import os


def make_folder(name):
    path = os.path.join("results", name)
    os.makedirs(path)
    return path