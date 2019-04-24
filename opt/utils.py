from __future__ import division

import os
import skopt
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from config import Configuration as config


def save_skopt_result(obj):
    f = os.path.join(
        config.EXPERIMENT_PERSISTENCE, 'bayesOptResults.sav')
    skopt.dump(obj, open(f, 'wb'))
    print('I: [save_skopt_result] obj saved successfully into {}'.format(f))


def save_txt(obj, filename):
    f = os.path.join(config.BO_RUN_PERSISTENCE, filename)
    file = open(f, 'w')
    file.write(obj)
    file.close()


def plot_cumulative_distribution(revisions):
    Y_3000s = []
    for revision, title in revisions:  # FIXME
        label = title  # FIXME
        # res = skopt.load('../generated/5.5/bayesOptResults.0.1.' + str(revision) + '-' + title + '.sav')
        res = skopt.load('../experiments/bayesOptResults.0.1.sav')
        func_vals = res.func_vals
        Y_3000s.append((-np.array([i for i in func_vals]), label))

    color = plt.cm.viridis(np.linspace(0, 4, 18))
    plt.rc('axes', prop_cycle=(cycler('color', color)))  # +
                               # cycler('linestyle', ['-', '--', ':', '-.'])))

    fig = plt.gcf()
    for data, label in Y_3000s:
        values, base = np.histogram(data, bins=40)
        cumulative = np.cumsum(values) / 60
        plt.plot(base[:-1], cumulative, linewidth=4, label=label)

    plt.legend()
    plt.grid()
    plt.tick_params(top=True, direction='in')
    plt.grid(which='major', linestyle='--', alpha=0.4)
    plt.show()

    return fig
