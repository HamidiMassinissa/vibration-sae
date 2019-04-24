import os
import numpy as np
from config import Configuration as config


def save_txt(obj, filename):
    f = os.path.join(config.BO_RUN_PERSISTENCE, filename)
    np.save(f, obj)
