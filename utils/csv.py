import csv


def write_detections_summary(output_path: str, video_path: str, nums, ntrajs, averages):
    """
    Outputted csv be like:
    Frame #    ID    Detection   Box W,L
    :param output_path:
    :param video_path:
    :param video_counter:
    :param nums:
    :return:
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([video_path, " "])
        writer.writerow(["Frame #", "ID (tracklet-specific)", "Detection (Center)", "Bounding Box (Width,Height)"])
        for frame in nums:
            framenum = frame
            ind = nums.index(frame)
            for t in ntrajs:
                det = t[ind]
                detid = ntrajs.index(t)
                distances = averages[detid]
                width = distances[0] + distances[0]
                length = distances[1] + distances[1]
                dim = "(" + str(width) + "," + str(length) + ")"
                writer.writerow([framenum, detid, det, dim])
