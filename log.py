# for logging
import json
import os
import time


def make_folder():
    timestr = time.strftime("date:%m_%d-time:%H_%M_%S")
    path = os.path.join("results", timestr)
    os.makedirs(path)
    return path


def save_dict(d, save_path, name):
    with open(os.path.join(save_path, name), "w") as f:
        json.dump(d, f, indent=4)
