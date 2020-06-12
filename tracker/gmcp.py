import os
import numpy
import itertools
import subprocess
import random
from PIL import Image
import pickle
import imageio
import matlab.engine
import cv2

from tracker.math import get_2D_dist
from utils.files import truncate_file, del_files
from utils.helpers import chunks
from utils.logger import logger
from utils.csv import write_detections_summary
from utils.yaml import read_cfg
from utils.parser import get_parser
from utils.colors import get_random_rgb
from utils.videos import merge_videos
from detectors.ssd.ssd import find_heads_ssd


FELZENSZWALB_PATH = "detectors/felzenszwalb/"
SEGMENT_VIDEOS_PATH = "fixtures/tmp/videos/segments/"


class GMCP:
    """
    GMCP-Tracker implementation
    """

    def __init__(self, input_video: str, output_video: str, tracklet_csv: str):
        self.video_in = input_video
        self.video_out = output_video
        self.tracklet_csv = tracklet_csv
        self.colors = [get_random_rgb() for x in range(0, 100)]

    def track(self, detector: str, detector_cfg: dict, frames_in_segment: int, frames_per_detection: int):
        assert detector in ["felzenszwalb", "ssd", "yolo"]
        detector_cfg = detector_cfg[detector]

        cap = cv2.VideoCapture(self.video_in)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_cut_frame_count = int(frame_count // frames_in_segment * frames_in_segment)
        logger.info(f"Number of frames detected: {int(frame_count)}")
        logger.info(f"Number of frames to be analysed: {video_cut_frame_count}")

        blank_image = Image.new("RGB", (5760, 2160))
        video = imageio.get_reader(self.video_in, 'ffmpeg')

        segment_counter = 0

        del_files(dir_path=SEGMENT_VIDEOS_PATH)
        truncate_file(filename=self.tracklet_csv)

        for x in range(0, int(frame_count), frames_in_segment):
            segment_counter += 1
            begval = x
            frames_in_segment = frames_per_detection
            hypdist = 80
            nums = [begval,
                    begval + frames_in_segment,
                    begval + frames_in_segment * 2,
                    begval + frames_in_segment * 3,
                    begval + frames_in_segment * 4,
                    begval + frames_in_segment * 5]
            framearray = []
            clusterarray = []
            clusterpoints = []
            indexcounter = 0
            histogramarray = []

            drframes = []

            donehyp = False
            points_hist = []
            points_cluster = []
            points_frame = []
            points_box = []
            boundingpoints = []
            boxesarray = []
            if detector == "felzenszwalb":
                eng = matlab.engine.start_matlab()
                eng.cd(FELZENSZWALB_PATH)

            for num in nums:
                image = video.get_data(num)
                index1 = nums.index(num)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if indexcounter == 0:
                    firstframe = image
                if indexcounter == 5:
                    lastframe = image
                cv2.imwrite('fixtures/tmp/img/saveim.jpeg', image)
                im = image.copy()
                image = "/fixtures/tmp/img/saveim.jpeg"
                himage = cv2.imread('fixtures/tmp/img/saveim.jpeg')
                path = os.getcwd()
                img_to_detect_on = str(path) + str(image)
                if detector == "felzenszwalb":
                    val = eng.persontest({'arg1': img_to_detect_on})  # na tym saveim.jpeg wykonywana jest detekcja
                    # tu trzeba zobaczyć co jest zwracane w val
                if detector == "ssd":
                    val = find_heads_ssd()

                counter = 0
                newlist = []
                personlist = []
                for s in val:
                    for c in s:
                        counter = counter + 1
                        if counter < 37:
                            c = round(c, 2)
                            newlist.append(c)
                    personlist.append(newlist)
                    newlist = []
                    counter = 0

                newpersonlist = []
                for s in personlist:
                    newperson = list(chunks(s, 4))
                    newpersonlist.append(newperson)

                bcounter = 0
                histarray = []
                bighist = []
                pointy = []
                arrayofpoints = []
                for person in newpersonlist:
                    logger.info(f"Object coordinates: +{str(person[0])}")
                    personcoord = person[0]
                    x1 = personcoord[0]
                    y1 = personcoord[1]
                    x2 = personcoord[2]
                    y2 = personcoord[3]
                    xpoint = (x1 + x2) / 2
                    ypoint = (y1 + y2) / 2
                    pointy.append(xpoint)
                    pointy.append(ypoint)
                    boundbox = [[x1, y1], [x2, y2]]
                    boundingpoints.append(boundbox)
                    clusterpoints.append(pointy)
                    pointy = []
                    if index1 == 0:
                        xpoint = xpoint
                        ypoint = ypoint
                    if index1 == 1:
                        xpoint = xpoint + 1920
                    if index1 == 2:
                        xpoint = xpoint + 3840
                    if index1 == 3:
                        ypoint = ypoint + 1080
                    if index1 == 4:
                        ypoint = ypoint + 1080
                        xpoint = xpoint + 1920
                    if index1 == 5:
                        ypoint = ypoint + 1080
                        xpoint = xpoint + 3840
                    pointy.append(xpoint)
                    pointy.append(ypoint)
                    arrayofpoints.append(pointy)
                    pointy = []
                    for body in person:
                        bcounter = bcounter + 1
                        if bcounter > 1:
                            logger.info(f"Found object with coordinates: {body}")
                            x1 = int(body[0])
                            y1 = int(body[1])
                            x2 = int(body[2])
                            y2 = int(body[3])

                            cropbody = himage[y1:y2, x1:x2]
                            cropbody = cv2.cvtColor(cropbody, cv2.COLOR_BGR2RGB)
                            try:
                                hist4 = cv2.calcHist(cropbody, [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                                hist4 = cv2.normalize(hist4, None).flatten()
                            except Exception as e:
                                logger.error(e)
                                hist4 = []
                                hist4 = numpy.array(hist4)
                            histarray.append(hist4)
                    bighist.append(histarray)
                    histarray = []
                    bcounter = 0

                framearray.append(arrayofpoints)
                boxesarray.append(boundingpoints)
                clusterarray.append(clusterpoints)
                histogramarray.append(bighist)
                clusterpoints = []
                boundingpoints = []

                logger.info(("finished image " + str(indexcounter)))
                img = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                indexcounter = indexcounter + 1

                pilimg = Image.fromarray(img)
                index = nums.index(num)
                blank_image.paste(pilimg, (1920 * index, 0))

                if index >= 3:
                    nloc = index - 3
                    blank_image.paste(pilimg, (1920 * nloc, 1080))

            for f in framearray:
                for p in f:
                    points_frame.append(p)

            for c in clusterarray:
                for p in c:
                    points_cluster.append(p)

            for b in boxesarray:
                for bp in b:
                    points_box.append(bp)

            safeties = [[0, 0], [450, 0], [900, 0], [0, 800], [450, 800], [900, 800]]
            counterhist = 0
            for h in histogramarray:
                for c in h:
                    counterhist = counterhist + 1
                    if counterhist == 1:
                        safetyhist1 = c
                    if counterhist == 2:
                        safetyhist2 = c
                    if counterhist == 3:
                        safetyhist3 = c
                    if counterhist == 4:
                        safetyhist4 = c
                    if counterhist == 5:
                        safetyhist5 = c
                    if counterhist == 6:
                        safetyhist6 = c
                    points_hist.append(c)

            # create input graph
            oblank = numpy.array(blank_image)
            oblank = cv2.cvtColor(oblank, cv2.COLOR_RGB2BGR)
            oblank = numpy.array(oblank)
            cv2.line(oblank, (10, 10), (100, 100), (255, 0, 0), 1)
            logger.info(framearray)
            logger.info("length of framearray begin")
            logger.info((len(framearray)))
            framecopy1 = list(framearray)

            outfile = open('fixtures/tmp/frame.pkl', 'wb')
            pickle.dump(framecopy1, outfile)
            outfile.close()
            tracklets1 = []
            framees = []
            hypotnodes = []
            poslengths = []
            while True:

                try:
                    f0 = random.choice(framearray[0])
                    index = points_frame.index(f0)
                    c0 = points_cluster[index]
                    f1 = random.choice(framearray[1])
                    index = points_frame.index(f1)
                    c1 = points_cluster[index]
                    f2 = random.choice(framearray[2])
                    index = points_frame.index(f2)
                    c2 = points_cluster[index]
                    f3 = random.choice(framearray[3])
                    index = points_frame.index(f3)
                    c3 = points_cluster[index]
                    f4 = random.choice(framearray[4])
                    index = points_frame.index(f4)
                    c4 = points_cluster[index]
                    f5 = random.choice(framearray[5])
                    index = points_frame.index(f5)
                    c5 = points_cluster[index]

                    arr = [c0, c1, c2, c3, c4, c5]

                    logger.info(("Inputted array " + str(arr)))

                    def minimumclique(arr, rcount):
                        distancep = 0
                        toedit = list(arr)

                        arrayofdistances1 = []
                        arrayofarrays = []

                        arrayofarrays.append(toedit)
                        array = toedit
                        for point in array:
                            curpoint = point
                            index = points_cluster.index(curpoint)
                            # caluclate real cluster point
                            hist = points_hist[index]
                            for p in array:
                                if p != curpoint:
                                    index4 = array.index(curpoint)
                                    if index4 == 0:
                                        if p in hypotnodes:
                                            distancep = distancep + 1000
                                        pindex = array.index(p)
                                        npindex = pindex - 1
                                        pp = array[npindex]
                                        index = points_cluster.index(p)
                                        # caluclate real cluster point
                                        hist2 = points_hist[index]

                                        # histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...

                                        try:
                                            d = cv2.compareHist(hist[0], hist2[0], cv2.HISTCMP_BHATTACHARYYA)
                                            distancep = distancep + d * 400

                                            d = cv2.compareHist(hist[1], hist2[1], cv2.HISTCMP_BHATTACHARYYA)
                                            distancep = distancep + d * 400

                                            d = cv2.compareHist(hist[2], hist2[2], cv2.HISTCMP_BHATTACHARYYA)
                                            distancep = distancep + d * 400

                                            d = cv2.compareHist(hist[3], hist2[3], cv2.HISTCMP_BHATTACHARYYA)
                                            distancep = distancep + d * 400

                                            d = cv2.compareHist(hist[4], hist2[4], cv2.HISTCMP_BHATTACHARYYA)
                                            distancep = distancep + d * 400

                                            d = cv2.compareHist(hist[5], hist2[5], cv2.HISTCMP_BHATTACHARYYA)
                                            distancep = distancep + d * 400

                                            d = cv2.compareHist(hist[6], hist2[6], cv2.HISTCMP_BHATTACHARYYA)
                                            distancep = distancep + d * 400

                                            d = cv2.compareHist(hist[7], hist2[7], cv2.HISTCMP_BHATTACHARYYA)
                                            distancep = distancep + d * 400
                                        except Exception as e:
                                            logger.error(e)
                                            continue

                                        distancep = distancep + get_2D_dist(pp[0], pp[1], p[0], p[1])

                        entiredistance = distancep
                        firstcost = entiredistance

                        arrayofdistances1.append(entiredistance)
                        distancep = 0

                        for val in framearray[0]:
                            toedit = list(arr)
                            index = points_frame.index(val)
                            v = points_cluster[index]

                            toedit[0] = v

                            arrayofarrays.append(toedit)

                            # calculate clique
                            array = toedit
                            for point in array:
                                curpoint = point
                                index = points_cluster.index(curpoint)
                                # caluclate real cluster point
                                hist = points_hist[index]
                                for p in array:
                                    if p != curpoint:
                                        index4 = array.index(curpoint)
                                        if index4 == 0:
                                            if p in hypotnodes:
                                                distancep = distancep + 1000
                                            pindex = array.index(p)
                                            npindex = pindex - 1
                                            pp = array[npindex]
                                            index = points_cluster.index(p)
                                            # caluclate real cluster point
                                            hist2 = points_hist[index]

                                            # histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...

                                            try:
                                                d = cv2.compareHist(hist[0], hist2[0], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[1], hist2[1], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[2], hist2[2], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[3], hist2[3], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[4], hist2[4], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[5], hist2[5], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[6], hist2[6], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[7], hist2[7], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400
                                            except:
                                                continue

                                            distancep = distancep + get_2D_dist(pp[0], pp[1], p[0], p[1])

                            entiredistance = distancep
                            logger.info(f"Cost: {str(entiredistance)}")
                            arrayofdistances1.append(entiredistance)
                            distancep = 0

                        for val in framearray[1]:
                            toedit = list(arr)
                            index = points_frame.index(val)
                            v = points_cluster[index]

                            toedit[1] = v
                            arrayofarrays.append(toedit)

                            # calculate clique
                            array = toedit
                            for point in array:
                                curpoint = point
                                index = points_cluster.index(curpoint)
                                # caluclate real cluster point
                                hist = points_hist[index]
                                for p in array:
                                    if p != curpoint:
                                        index4 = array.index(curpoint)
                                        if index4 == 0:
                                            if p in hypotnodes:
                                                distancep = distancep + 1000
                                            pindex = array.index(p)
                                            npindex = pindex - 1
                                            pp = array[npindex]
                                            index = points_cluster.index(p)
                                            # caluclate real cluster point
                                            hist2 = points_hist[index]

                                            # histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...

                                            try:
                                                d = cv2.compareHist(hist[0], hist2[0], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[1], hist2[1], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[2], hist2[2], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[3], hist2[3], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[4], hist2[4], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[5], hist2[5], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[6], hist2[6], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[7], hist2[7], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400
                                            except:
                                                continue

                                            distancep = distancep + get_2D_dist(pp[0], pp[1], p[0], p[1])

                            entiredistance = distancep
                            arrayofdistances1.append(entiredistance)
                            distancep = 0

                        for val in framearray[2]:
                            toedit = list(arr)
                            index = points_frame.index(val)
                            v = points_cluster[index]
                            toedit[2] = v
                            arrayofarrays.append(toedit)

                            # calculate clique
                            array = toedit
                            for point in array:
                                curpoint = point
                                index = points_cluster.index(curpoint)
                                # caluclate real cluster point
                                hist = points_hist[index]
                                for p in array:
                                    if p != curpoint:
                                        index4 = array.index(curpoint)
                                        if index4 == 0:
                                            if p in hypotnodes:
                                                distancep = distancep + 1000
                                            pindex = array.index(p)
                                            npindex = pindex - 1
                                            pp = array[npindex]
                                            index = points_cluster.index(p)
                                            # caluclate real cluster point
                                            hist2 = points_hist[index]

                                            # histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...

                                            try:
                                                d = cv2.compareHist(hist[0], hist2[0], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[1], hist2[1], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[2], hist2[2], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[3], hist2[3], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[4], hist2[4], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[5], hist2[5], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[6], hist2[6], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[7], hist2[7], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400
                                            except:
                                                continue

                                            distancep = distancep + get_2D_dist(pp[0], pp[1], p[0], p[1])

                            entiredistance = distancep
                            arrayofdistances1.append(entiredistance)
                            distancep = 0

                        for val in framearray[3]:
                            toedit = list(arr)
                            index = points_frame.index(val)
                            v = points_cluster[index]
                            toedit[3] = v
                            arrayofarrays.append(toedit)

                            # calculate clique
                            array = toedit
                            for point in array:
                                curpoint = point
                                index = points_cluster.index(curpoint)
                                # caluclate real cluster point
                                hist = points_hist[index]
                                for p in array:
                                    if p != curpoint:
                                        index4 = array.index(curpoint)
                                        if index4 == 0:
                                            if p in hypotnodes:
                                                distancep = distancep + 1000
                                            pindex = array.index(p)
                                            npindex = pindex - 1
                                            pp = array[npindex]
                                            index = points_cluster.index(p)
                                            # caluclate real cluster point
                                            hist2 = points_hist[index]

                                            # histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...

                                            try:
                                                d = cv2.compareHist(hist[0], hist2[0], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[1], hist2[1], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[2], hist2[2], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[3], hist2[3], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[4], hist2[4], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[5], hist2[5], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[6], hist2[6], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[7], hist2[7], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400
                                            except:
                                                continue

                                            distancep = distancep + get_2D_dist(pp[0], pp[1], p[0], p[1])

                            entiredistance = distancep
                            arrayofdistances1.append(entiredistance)
                            distancep = 0

                        for val in framearray[4]:
                            toedit = list(arr)
                            index = points_frame.index(val)
                            v = points_cluster[index]

                            toedit[4] = v
                            arrayofarrays.append(toedit)

                            # calculate clique
                            array = toedit
                            for point in array:
                                curpoint = point
                                index = points_cluster.index(curpoint)
                                # caluclate real cluster point
                                hist = points_hist[index]
                                for p in array:
                                    if p != curpoint:
                                        index4 = array.index(curpoint)
                                        if index4 == 0:
                                            if p in hypotnodes:
                                                distancep = distancep + 1000
                                            pindex = array.index(p)
                                            npindex = pindex - 1
                                            pp = array[npindex]
                                            index = points_cluster.index(p)
                                            # caluclate real cluster point
                                            hist2 = points_hist[index]

                                            # histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...

                                            try:
                                                d = cv2.compareHist(hist[0], hist2[0], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[1], hist2[1], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[2], hist2[2], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[3], hist2[3], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[4], hist2[4], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[5], hist2[5], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[6], hist2[6], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[7], hist2[7], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400
                                            except:
                                                continue

                                            distancep = distancep + get_2D_dist(pp[0], pp[1], p[0], p[1])

                            entiredistance = distancep
                            arrayofdistances1.append(entiredistance)
                            distancep = 0

                        for val in framearray[5]:
                            toedit = list(arr)
                            index = points_frame.index(val)
                            v = points_cluster[index]

                            toedit[5] = v
                            arrayofarrays.append(toedit)

                            # calculate clique
                            array = toedit
                            for point in array:
                                curpoint = point
                                index = points_cluster.index(curpoint)
                                # caluclate real cluster point
                                hist = points_hist[index]
                                for p in array:
                                    if p != curpoint:
                                        index4 = array.index(curpoint)
                                        if index4 == 0:
                                            if p in hypotnodes:
                                                distancep = distancep + 1000
                                            pindex = array.index(p)
                                            npindex = pindex - 1
                                            pp = array[npindex]
                                            index = points_cluster.index(p)
                                            # caluclate real cluster point
                                            hist2 = points_hist[index]

                                            # histogram intersection between hist[0] and hist2[0], hist[1] and hist2[1], ...

                                            try:
                                                d = cv2.compareHist(hist[0], hist2[0], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[1], hist2[1], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[2], hist2[2], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[3], hist2[3], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[4], hist2[4], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[5], hist2[5], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[6], hist2[6], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400

                                                d = cv2.compareHist(hist[7], hist2[7], cv2.HISTCMP_BHATTACHARYYA)
                                                distancep = distancep + d * 400
                                            except:
                                                continue

                                            distancep = distancep + get_2D_dist(pp[0], pp[1], p[0], p[1])

                            entiredistance = distancep
                            arrayofdistances1.append(entiredistance)
                            distancep = 0
                        minimum = min(arrayofdistances1)

                        print(("FIRST COST:" + str(firstcost)))
                        print(("MINIMUM:" + str(minimum)))

                        index = arrayofdistances1.index(minimum)

                        minimum_path = arrayofarrays[index]

                        print(("CHOSEN PATH: " + str(minimum_path)))

                        dif = firstcost - minimum
                        rcount = rcount + 1
                        if dif < 100 or rcount == 5:
                            print(("returning minimum_path " + str(minimum_path)))
                            return minimum_path
                        else:
                            print(("calling again with minimum_path " + str(minimum_path)))
                            return minimumclique(minimum_path, rcount)

                    rcount = 0
                    minimum_clique = minimumclique(arr, rcount)

                    logger.info(minimum_clique)

                    minimum_path = minimum_clique

                    logger.info(donehyp)
                    logger.info(minimum_path)

                    seed = minimum_path[0]
                    point1 = minimum_path[1]
                    point2 = minimum_path[2]
                    point3 = minimum_path[3]
                    point4 = minimum_path[4]
                    point5 = minimum_path[5]

                    pointsfortrack = []
                    pointsfortrack.append(seed)
                    pointsfortrack.append(point1)
                    pointsfortrack.append(point2)
                    pointsfortrack.append(point3)
                    pointsfortrack.append(point4)
                    pointsfortrack.append(point5)

                    tracklets1.append(pointsfortrack)

                    index = points_cluster.index(seed)
                    # caluclate real cluster point
                    seed = points_frame[index]

                    index = points_cluster.index(point1)
                    # caluclate real cluster point
                    point1 = points_frame[index]

                    index = points_cluster.index(point2)
                    # caluclate real cluster point
                    point2 = points_frame[index]

                    index = points_cluster.index(point3)
                    # caluclate real cluster point
                    point3 = points_frame[index]

                    index = points_cluster.index(point4)
                    # caluclate real cluster point
                    point4 = points_frame[index]

                    index = points_cluster.index(point5)
                    # caluclate real cluster point
                    point5 = points_frame[index]

                    logger.info(minimum_path)
                    clist = minimum_path
                    d1 = get_2D_dist(clist[0][0], clist[0][1], clist[1][0], clist[1][1])

                    d2 = get_2D_dist(clist[1][0], clist[1][1], clist[2][0], clist[2][1])

                    d3 = get_2D_dist(clist[2][0], clist[2][1], clist[3][0], clist[3][1])

                    d4 = get_2D_dist(clist[3][0], clist[3][1], clist[4][0], clist[4][1])

                    d5 = get_2D_dist(clist[4][0], clist[4][1], clist[5][0], clist[5][1])

                    cslopes = [d1, d2, d3, d4, d5]

                    sureshots = []
                    for s in cslopes:
                        if s > hypdist:
                            for i, val in enumerate(cslopes):
                                if s == val:
                                    sureshots.append(i)
                    logger.info(sureshots)
                    surehypots = []
                    if sureshots != []:
                        if len(sureshots) == 1:
                            s = sureshots[0]
                            surehypots = minimum_path[s + 1]
                            surehypots = [surehypots]
                        else:
                            beg = sureshots[0]
                            beg = beg + 1
                            last = sureshots[-1]
                            last = last + 1
                            if beg == 5:
                                surehypots = minimum_path[-1]
                            if beg == 4 and last == 5:
                                logger.info("last 2")
                                surehypots = minimum_path[-2:]
                                surehypots = surehypots
                            else:
                                surehypots = minimum_path[beg:last]
                        logger.info(("Sureshot hypothetical node required here: " + str(surehypots)))
                    toremove = []

                    indexes = []
                    logger.info("length of framearray")
                    logger.info((len(framearray)))
                    for array in framearray:
                        for point in array:
                            if point == seed:
                                i1 = framearray.index(array)
                                i2 = array.index(point)
                                indexes.append(i1)
                                indexes.append(i2)
                                toremove.append(indexes)
                                indexes = []
                            if point == point1:
                                i1 = framearray.index(array)
                                i2 = array.index(point)
                                indexes.append(i1)
                                indexes.append(i2)
                                toremove.append(indexes)
                                indexes = []
                            if point == point2:
                                i1 = framearray.index(array)
                                i2 = array.index(point)
                                indexes.append(i1)
                                indexes.append(i2)
                                toremove.append(indexes)
                                indexes = []
                            if point == point3:
                                i1 = framearray.index(array)
                                i2 = array.index(point)
                                indexes.append(i1)
                                indexes.append(i2)
                                toremove.append(indexes)
                                indexes = []
                            if point == point4:
                                i1 = framearray.index(array)
                                i2 = array.index(point)
                                indexes.append(i1)
                                indexes.append(i2)
                                toremove.append(indexes)
                                indexes = []
                            if point == point5:
                                i1 = framearray.index(array)
                                i2 = array.index(point)
                                indexes.append(i1)
                                indexes.append(i2)
                                toremove.append(indexes)
                                indexes = []
                    logger.info(toremove)
                    toremove2 = list(toremove)
                    toremove.sort()
                    allpos2 = list(toremove2 for toremove2, _ in itertools.groupby(toremove2))
                    toremove = allpos2
                    for indexes in toremove:
                        i1 = indexes[0]
                        i2 = indexes[1]
                        try:
                            point = framearray[i1][i2]
                            ind = points_frame.index(point)
                            cpoint = points_cluster[ind]
                            if cpoint in surehypots:
                                if cpoint in drframes:
                                    try:
                                        logger.info("removing " + str(framearray[i1][i2]))
                                        del framearray[i1][i2]
                                    except:
                                        pass
                                else:
                                    logger.info("found a saver " + str(cpoint))
                                    pass
                            else:
                                try:
                                    logger.info("removing " + str(framearray[i1][i2]))
                                    del framearray[i1][i2]
                                except:
                                    logger.error("unable to remove " + str(framearray[i1][i2]))
                        except:
                            logger.error("houston we got problem")
                    logger.info(len(framearray))
                    logger.info(framearray)

                    for i, l in enumerate(poslengths):
                        if i > 3:
                            l1 = poslengths[i]
                            l2 = poslengths[i - 1]
                            l3 = poslengths[i - 2]

                            if l1 == l2 and l2 == l3:
                                logger.warn("endless loop detected")
                                for indexes in toremove:
                                    i1 = indexes[0]
                                    i2 = indexes[1]
                                    del framearray[i1][i2]
                                break

                    logger.info("before")
                    logger.info(donehyp)
                    empty = False
                    for frame in framearray:
                        if not frame:
                            empty = True
                    s = []
                    if empty and donehyp == False:
                        for frame in framearray:
                            if frame:
                                fl = len(frame)
                                s.append(fl)
                        avg = sum(s) / len(s)
                        avg = int(avg)
                        avg = avg + 1
                        logger.info(avg)
                        for i, frame in enumerate(framearray):
                            if frame == []:
                                for x in range(0, avg):
                                    if i == 0:
                                        framearray[i].append([0, x])
                                        framees.append([i, [0, x]])
                                        drframes.append([0, x])
                                        points_frame.append([0, x])
                                        points_cluster.append([0, x])
                                        points_hist.append(safetyhist1)
                                    if i == 1:
                                        framearray[i].append([2370, x])
                                        framees.append([i, [2370, x]])
                                        points_frame.append([2370, x])
                                        points_cluster.append([450, x])
                                        drframes.append([450, x])
                                        points_hist.append(safetyhist1)
                                    if i == 2:
                                        framees.append([i, [4740, x]])
                                        framearray[i].append([4740, x])
                                        points_frame.append([4740, x])
                                        points_cluster.append([900, x])
                                        drframes.append([900, x])
                                        points_hist.append(safetyhist3)
                                    if i == 3:
                                        framees.append([i, [0, 1880 + x]])
                                        framearray[i].append([0, 1880 + x])
                                        points_frame.append([0, 1880 + x])
                                        points_cluster.append([0, 800 + x])
                                        drframes.append([0, 800 + x])
                                        points_hist.append(safetyhist4)
                                    if i == 4:
                                        framees.append([i, [2370, 1880 + x]])
                                        framearray[i].append([2370, 1880 + x])
                                        points_frame.append([2370, 1880 + x])
                                        points_cluster.append([450, 800 + x])
                                        drframes.append([450, 800 + x])
                                        points_hist.append(safetyhist5)
                                    if i == 5:
                                        framees.append([i, [4740, 1880 + x]])
                                        framearray[i].append([4740, 1880 + x])
                                        points_frame.append([4740, 1880 + x])
                                        points_cluster.append([900, 800 + x])
                                        drframes.append([900, 800 + x])
                                        points_hist.append(safetyhist6)

                        donehyp = True

                except Exception as e:
                    logger.error(str(e))
                    break

            oblank = cv2.cvtColor(oblank, cv2.COLOR_RGB2BGR)
            lined = Image.fromarray(oblank)

            f = open('fixtures/tmp/frame.pkl', 'rb')
            framecopy2 = pickle.load(f)

            framecopy2[0].append([0, 0])
            framecopy2[1].append([1920, 0])
            framecopy2[2].append([3840, 0])
            framecopy2[3].append([0, 1080])
            framecopy2[4].append([1920, 1080])
            framecopy2[5].append([3840, 1080])

            for f in framees:
                if f[0] == 0:
                    framecopy2[0].append(f[1])
                if f[0] == 1:
                    framecopy2[1].append(f[1])
                if f[0] == 2:
                    framecopy2[2].append(f[1])
                if f[0] == 3:
                    framecopy2[3].append(f[1])
                if f[0] == 4:
                    framecopy2[4].append(f[1])
                if f[0] == 5:
                    framecopy2[5].append(f[1])

            firstframe = Image.fromarray(firstframe)
            lastframe = Image.fromarray(lastframe)
            blended = Image.blend(lastframe, firstframe, 0)
            blended = numpy.array(blended)

            safteycounter = 0
            poplist = []
            for track in tracklets1:
                for point in track:
                    if point in safeties:
                        safteycounter = safteycounter + 1
                index = tracklets1.index(track)
                if safteycounter > 3:
                    poplist.append(index)
                safteycounter = 0

            for i in poplist:
                tracklets1.pop(i)

            list1 = [[0, 0], [450, 0], [900, 0], [0, 800], [450, 800], [900, 800]]

            if list1 in tracklets1:
                tracklets1.remove(list1)

            alltracks = []
            for tracklet in tracklets1:
                alltracks.append(tracklet)

            blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

            trajs = alltracks

            for t in trajs:
                logger.info(t)

            def hypothetical(trajectory, cin):
                ntrajs = []
                trajs = trajectory
                for tracklet in trajs:
                    etracklet = list(tracklet)
                    logger.info(tracklet)
                    clist = tracklet
                    d1 = get_2D_dist(clist[0][0], clist[0][1], clist[1][0], clist[1][1])

                    d2 = get_2D_dist(clist[1][0], clist[1][1], clist[2][0], clist[2][1])

                    d3 = get_2D_dist(clist[2][0], clist[2][1], clist[3][0], clist[3][1])

                    d4 = get_2D_dist(clist[3][0], clist[3][1], clist[4][0], clist[4][1])

                    d5 = get_2D_dist(clist[4][0], clist[4][1], clist[5][0], clist[5][1])

                    cslopes = [d1, d2, d3, d4, d5]

                    sureshots = []
                    for s in cslopes:
                        if s > hypdist:
                            for i, val in enumerate(cslopes):
                                if s == val:
                                    sureshots.append(i)

                    logger.info(sureshots)
                    surehypots = []
                    if sureshots != []:
                        if len(sureshots) == 1:
                            s = sureshots[0]
                            surehypots = tracklet[s + 1]
                            surehypots = [surehypots]
                        else:
                            beg = sureshots[0]
                            beg = beg + 1
                            last = sureshots[-1]
                            last = last + 1
                            if beg == 5:
                                surehypots = tracklet[-1]
                            if beg == 4 and last == 5:
                                logger.info("last 2")
                                surehypots = tracklet[-2:]
                                surehypots = surehypots
                            else:
                                surehypots = tracklet[beg:last]
                        logger.info("Sureshot hypothetical node required here: " + str(surehypots))
                    for hyp in surehypots:
                        etracklet.remove(hyp)
                    logger.info("Inliers: " + str(etracklet))
                    logger.info("Outliers: " + str(surehypots))
                    for hyp in surehypots:
                        index = tracklet.index(hyp)
                        if len(etracklet) < 3:
                            val1 = etracklet[0]
                            val2 = etracklet[1]
                            changex = val2[0] - val1[0]
                            changey = val2[1] - val1[1]
                        else:
                            try:
                                if index - 2 < 0 or index - 1 < 0:
                                    raise ValueError('A very specific bad thing happened')
                                val1 = etracklet[index - 2]
                                val2 = etracklet[index - 1]
                                changex = val2[0] - val1[0]
                                changey = val2[1] - val1[1]
                            except:
                                val1 = etracklet[index]
                                val2 = etracklet[index + 1]
                                changex = val2[0] - val1[0]
                                changey = val2[1] - val1[1]
                        try:
                            samp = etracklet[index - 1]
                            hypotx = samp[0] + changex
                            hypoty = samp[1] + changey
                            hypot = [hypotx, hypoty]
                        except:
                            samp = val1
                            hypotx = samp[0] - changex
                            hypoty = samp[1] - changey
                            hypot = [hypotx, hypoty]
                        logger.info("inserting " + str(hypot))
                        etracklet.insert(index, hypot)
                    logger.info(etracklet)
                    ntrajs.append(etracklet)
                if cin == 6:
                    return ntrajs
                if cin < 6:
                    cin = cin + 1
                    return hypothetical(ntrajs, cin)

            logger.info("DISTANCES")

            ntrajs = hypothetical(trajs, 0)
            logger.info(ntrajs)
            for tracklet in ntrajs:
                logger.info(tracklet)

            mtrajs = []
            for tracklet in ntrajs:
                seed = tracklet[0]
                point1 = tracklet[1]
                point2 = tracklet[2]
                point3 = tracklet[3]
                point4 = tracklet[4]
                point5 = tracklet[5]
                xaverage = (seed[0] + point1[0] + point2[0] + point3[0] + point4[0] + point5[0]) / 6
                yaverage = (seed[1] + point1[1] + point2[1] + point3[1] + point4[1] + point5[1]) / 6

                average = [xaverage, yaverage]
                mtrajs.append([seed, average, point5])

            buffer1 = hypdist + 20
            toremove = []
            for tracklet in ntrajs:
                nope = False
                clist = tracklet

                d1 = get_2D_dist(clist[0][0], clist[0][1], clist[1][0], clist[1][1])

                d2 = get_2D_dist(clist[1][0], clist[1][1], clist[2][0], clist[2][1])

                d3 = get_2D_dist(clist[2][0], clist[2][1], clist[3][0], clist[3][1])

                d4 = get_2D_dist(clist[3][0], clist[3][1], clist[4][0], clist[4][1])

                d5 = get_2D_dist(clist[4][0], clist[4][1], clist[5][0], clist[5][1])

                cslopes = [d1, d2, d3, d4, d5]
                for s in cslopes:
                    if s > buffer1:
                        nope = True
                if nope:
                    toremove.append(tracklet)

            for t in toremove:
                ntrajs.remove(t)

            toremove2 = []
            for tracklet in ntrajs:
                p1 = tracklet
                for t in ntrajs:
                    if t != p1:
                        p2 = t
                        if p1[0] == p2[0]:
                            p1counter = 0
                            p2counter = 0
                            p1points = []
                            p2points = []
                            for point in p1:
                                if point in points_cluster:
                                    if point in p1points:
                                        pass
                                    else:
                                        p1points.append(point)
                                        p1counter = p1counter + 1
                            for point in p2:
                                if point in points_cluster:
                                    if point in p2points:
                                        pass
                                    else:
                                        p2points.append(point)
                                        p2counter = p2counter + 1
                            if p1counter > p2counter:
                                toremove2.append(p2)
                            else:
                                pass

            for tr in toremove2:
                try:
                    ntrajs.remove(tr)
                except:
                    pass

            # removing duplicates
            try:
                ntrajs.sort()
                allpos = list(ntrajs for ntrajs, _ in itertools.groupby(ntrajs))
                ntrajs = allpos
            except Exception as e:
                logger.error(str(e))

            toremove3 = []
            for t in ntrajs:
                pcounter = 0
                for point in t:
                    if point in points_cluster:
                        if point in drframes:
                            pass
                        else:
                            pcounter = pcounter + 1
                if pcounter > 0:
                    pass
                else:
                    toremove3.append(t)

            for t in toremove3:
                ntrajs.remove(t)

            toremove4 = []

            for t in ntrajs:
                counterp = 0
                for point in t:
                    if point in points_cluster:
                        counterp = counterp + 1
                if counterp <= 1:
                    toremove4.append(t)
                if counterp == 2:
                    p1 = t[0]
                    for tp in ntrajs:
                        p2 = tp[0]
                        dist = get_2D_dist(p1[0], p1[1], p2[0], p2[1])
                        if dist < 10:
                            toremove4.append(t)

            for t in toremove4:
                try:
                    ntrajs.remove(t)
                except:
                    pass

            for t in ntrajs:
                logger.info(t)

            averages = []
            for tracklet in ntrajs:
                xdists = []
                ydists = []
                for point in tracklet:
                    if point in points_cluster:
                        index = points_cluster.index(point)
                        box = points_box[index]

                        p1 = box[0]
                        p2 = box[1]
                        x2 = p2[0]
                        y2 = p2[1]
                        xm = point[0]
                        ym = point[1]
                        xdist = x2 - xm
                        ydist = y2 - ym
                        xdists.append(xdist)
                        ydists.append(ydist)

                xaverage = sum(xdists) / len(xdists)
                yaverage = sum(ydists) / len(ydists)

                avgvector = []
                avgvector.append(xaverage)
                avgvector.append(yaverage)

                averages.append(avgvector)

            write_detections_summary(output_path=self.tracklet_csv,
                                     video_path=self.video_in,
                                     nums=nums,
                                     ntrajs=ntrajs,
                                     averages=averages)

            cap = cv2.VideoCapture(self.video_in)

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vw = cv2.VideoWriter('fixtures/tmp/videos/out.mp4', fourcc, 15, (w, h))
            counter = 0

            start = begval

            second = start + frames_in_segment
            third = start + frames_in_segment * 2
            fourth = start + frames_in_segment * 3
            fifth = start + frames_in_segment * 4
            sixth = start + frames_in_segment * 5
            last = start + frames_in_segment * 5

            while True:
                ret, frame = cap.read()
                counter = counter + 1
                if ret:
                    for trajectories in ntrajs:
                        index = ntrajs.index(trajectories)

                        distances = averages[index]

                        xdist = distances[0]
                        ydist = distances[1]

                        color = self.colors[index]

                        if counter == start:
                            cv2.rectangle(frame, (int(trajectories[0][0] - xdist), int(trajectories[0][1] - ydist)),
                                          (int(trajectories[0][0] + xdist), int(trajectories[0][1] + ydist)), color, 4)
                        if start < counter < second:
                            counter1 = counter - start
                            dist = second - start
                            curprog = counter1 / float(dist)
                            x1 = trajectories[0][0]
                            x2 = trajectories[1][0]
                            difx = x2 - x1
                            curx = float(difx * curprog)
                            curx = x1 + curx
                            y1 = trajectories[0][1]
                            y2 = trajectories[1][1]
                            dify = y2 - y1
                            cury = float(dify * curprog)
                            cury = y1 + cury
                            cv2.rectangle(frame, (int(curx - xdist), int(cury - ydist)),
                                          (int(curx + xdist), int(cury + ydist)), color, 4)
                            # cv2.imshow("frame",frame)
                            # cv2.waitKey(0)
                        if counter == second:
                            cv2.rectangle(frame, (int(trajectories[1][0] - xdist), int(trajectories[1][1] - ydist)),
                                          (int(trajectories[1][0] + xdist), int(trajectories[1][1] + ydist)), color, 4)
                        if second < counter < third:
                            counter2 = counter - second
                            dist = third - second
                            curprog = counter2 / float(dist)
                            x1 = trajectories[1][0]
                            x2 = trajectories[2][0]
                            difx = x2 - x1
                            curx = float(difx * curprog)
                            curx = x1 + curx
                            y1 = trajectories[1][1]
                            y2 = trajectories[2][1]
                            dify = y2 - y1
                            cury = float(dify * curprog)
                            cury = y1 + cury
                            cv2.rectangle(frame, (int(curx - xdist), int(cury - ydist)),
                                          (int(curx + xdist), int(cury + ydist)), color, 4)
                        if counter == third:
                            cv2.rectangle(frame, (int(trajectories[2][0] - xdist), int(trajectories[2][1] - ydist)),
                                          (int(trajectories[2][0] + xdist), int(trajectories[2][1] + ydist)), color, 4)
                        if third < counter < fourth:
                            counter3 = counter - third
                            dist = fourth - third
                            curprog = counter3 / float(dist)
                            x1 = trajectories[2][0]
                            x2 = trajectories[3][0]
                            difx = x2 - x1
                            curx = float(difx * curprog)
                            curx = x1 + curx
                            y1 = trajectories[2][1]
                            y2 = trajectories[3][1]
                            dify = y2 - y1
                            cury = float(dify * curprog)
                            cury = y1 + cury
                            cv2.rectangle(frame, (int(curx - xdist), int(cury - ydist)),
                                          (int(curx + xdist), int(cury + ydist)), color, 4)
                        if counter == fourth:
                            cv2.rectangle(frame, (int(trajectories[3][0] - xdist), int(trajectories[3][1] - ydist)),
                                          (int(trajectories[3][0] + xdist), int(trajectories[3][1] + ydist)), color, 4)
                        if fourth < counter < fifth:
                            counter4 = counter - fourth
                            dist = fifth - fourth
                            curprog = counter4 / float(dist)
                            x1 = trajectories[3][0]
                            x2 = trajectories[4][0]
                            difx = x2 - x1
                            curx = float(difx * curprog)
                            curx = x1 + curx
                            y1 = trajectories[3][1]
                            y2 = trajectories[4][1]
                            dify = y2 - y1
                            cury = float(dify * curprog)
                            cury = y1 + cury
                            cv2.rectangle(frame, (int(curx - xdist), int(cury - ydist)),
                                          (int(curx + xdist), int(cury + ydist)), color, 4)
                        if counter == fifth:
                            cv2.rectangle(frame, (int(trajectories[4][0] - xdist), int(trajectories[4][1] - ydist)),
                                          (int(trajectories[4][0] + xdist), int(trajectories[4][1] + ydist)), color, 4)
                        if fifth < counter < sixth:
                            counter5 = counter - fifth
                            dist = sixth - fifth
                            curprog = counter5 / float(dist)
                            x1 = trajectories[4][0]
                            x2 = trajectories[5][0]
                            difx = x2 - x1
                            curx = float(difx * curprog)
                            curx = x1 + curx
                            y1 = trajectories[4][1]
                            y2 = trajectories[5][1]
                            dify = y2 - y1
                            cury = float(dify * curprog)
                            cury = y1 + cury
                            cv2.rectangle(frame, (int(curx - xdist), int(cury - ydist)),
                                          (int(curx + xdist), int(cury + ydist)), color, 4)
                        if counter == sixth:
                            cv2.rectangle(frame, (int(trajectories[5][0] - xdist), int(trajectories[5][1] - ydist)),
                                          (int(trajectories[5][0] + xdist), int(trajectories[5][1] + ydist)), color, 4)

                    if counter >= start:
                        vw.write(frame)

                    if counter >= last:
                        break
                else:
                    pass

            cap.release()
            vw.release()
            cv2.destroyAllWindows()

            vidstring = f"ffmpeg -i fixtures/tmp/videos/out.mp4 {SEGMENT_VIDEOS_PATH}" + str(segment_counter) + ".mp4"
            subprocess.call(vidstring, shell=True)

        merge_videos(videos_dir=SEGMENT_VIDEOS_PATH+"*", output_file=self.video_out)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    cfg = read_cfg(args.conf_path)
    detector_cfg = cfg["detectors"]
    tracker_cfg = cfg["tracker"]

    tracker = GMCP(input_video=tracker_cfg["input_video"],
                   output_video=tracker_cfg["output_video"],
                   tracklet_csv=tracker_cfg["tracklet_detections"]). \
        track(detector=args.detector,
              detector_cfg=detector_cfg,
              frames_in_segment=tracker_cfg["frames_in_segment"],
              frames_per_detection=tracker_cfg["detection_per_frames"])
