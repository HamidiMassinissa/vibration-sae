import os
import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import sklearn.metrics

from models.utils import plot_loss


def save_figure(fig, title):
    path = os.path.join(
        config.BO_RUN_PERSISTENCE,
        '{}.svg'.format(title))
    plt.savefig(path, format='svg')
    plt.close(fig)
