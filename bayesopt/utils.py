import os
import skopt

from config import Configuration as config


def save_skopt_result(obj):
    f = os.path.join(
        config.EXPERIMENT_PERSISTENCE, 'bayesOptResults.sav')
    skopt.dump(obj, open(f, 'wb'))
    print('I: [save_skopt_result] obj saved successfully into {}'.format(f))
