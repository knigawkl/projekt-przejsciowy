import glob
import os


def truncate_file(filename: str):
    f = open(filename, "w+")
    f.close()


def del_files(dir_path: str):
    files = glob.glob(dir_path + "*")
    for f in files:
        os.remove(f)


def del_file(path: str):
    os.remove(path)


def sorted_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime))