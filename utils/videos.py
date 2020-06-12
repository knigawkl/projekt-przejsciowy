import glob
import subprocess

from tracker.math import sort_num
from utils.files import del_file


TMP_TXT = "files.txt"


def merge_videos(videos_dir: str, output_file: str):
    """
    Merge videos into one
    :param output_file: Where the merged video should be stored
    :param videos_dir: Path from where videos should be collected
    """
    files = glob.glob(videos_dir)
    sorted_files = sorted(files, key=sort_num)
    with open(TMP_TXT, "w") as text_file:
        for f in sorted_files:
            text_file.write("file \'" + f + "\'\n")
    subprocess.call(f"ffmpeg -f concat -safe 0 -i {TMP_TXT} -c copy {output_file} -y", shell=True)
    del_file(TMP_TXT)


def cut_video(video_path: str, frames_count: int, output_path: str):
    """
    Cuts video from given path and saves the output path
    :param output_path: where the cut video will be saved
    :param video_path: path to the video to be cut
    :param frames_count: how many frames from the beginning should be cut
    """
    subprocess.call(f"ffmpeg -i {video_path} -c:v libx264 -c:a aac -frames:v {frames_count} {output_path}")


# cut_video(video_path="/home/lk/Desktop/gmcp-tracker/fixtures/videos/input/pets3s.mp4",
#           frames_count=20,
#           output_path="out.mp4")

# merge_videos(videos_dir="/home/lk/Desktop/gmcp-tracker/fixtures/tmp/videos/segments/*",
#              output_file="/home/lk/Desktop/gmcp-tracker/fixtures/videos/output/out.mp4")
