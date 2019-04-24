import os
import numpy as np
from config import Configuration as config


def save_txt(obj, filename):
    f = os.path.join(config.BO_RUN_PERSISTENCE, filename)
    obj = np.array(obj)
    np.save(f, obj)
    return f


def load_txt(path):
    """ precondition: the complete path to the object """
    a = np.load(path)
    return a
