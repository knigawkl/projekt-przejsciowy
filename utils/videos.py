import glob
import subprocess

from tracker.math import sort_num


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

    subprocess.call(f"ffmpeg -f concat -safe 0 -i {TMP_TXT} -c copy {output_file}", shell=True)


merge_videos(videos_dir="/home/lk/Desktop/gmcp-tracker/fixtures/tmp/videos/*",
             output_file="/home/lk/Desktop/gmcp-tracker/fixtures/videos/output/out.mp4")
