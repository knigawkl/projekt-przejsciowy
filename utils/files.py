import glob
import os


def truncate_file(filename: str):
    f = open(filename, "w+")
    f.close()


def del_files(dir_path: str):
    files = glob.glob(dir_path + "*")
    for f in files:
        os.remove(f)
