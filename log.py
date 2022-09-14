import os
import time


def make_folder():
    timestr = time.strftime("date:%m_%d-time:%H_%M_%S")
    path = os.path.join("results", timestr)
    os.makedirs(path)
    return path