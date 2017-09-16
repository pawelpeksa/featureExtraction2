import os
import time

from os import listdir, makedirs
from os.path import isfile, join, exists

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def maybe_create_directory(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    @staticmethod
    def file_exist(path):
        return os.path.exists(path)        
        
    @staticmethod
    def get_seed():
        t = time.time() - int(time.time())
        t *= 1000000
        return int(t)        