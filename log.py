# for logging
import os
import time
import json

def make_folder():
    timestr = time.strftime("date:%m_%d-time:%H_%M_%S")
    os.mkdir(timestr)
    return timestr


def save_dict(d, save_path, name):
    with open(os.path.join(save_path, name), "w") as f:
        json.dump(d, f, indent=4)
