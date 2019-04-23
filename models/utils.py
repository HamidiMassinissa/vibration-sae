import os
import numpy as np
from deprecated import deprecated

from config import Configuration as config


@deprecated(version='0.1.0', reason='You should use save')
def save_txt(obj, filename):
    f = os.path.join(config.BO_RUN_PERSISTENCE, filename)
    file = open(f, 'w')
    file.write(obj)
    file.close()


def save(filename, what):
    f = os.path.join(config.BO_RUN_PERSISTENCE, filename)
    np.save(f, what)


def save_figure(fig, title):
    path = os.path.join(
        config.BO_RUN_PERSISTENCE,
        '{}.svg'.format(title))
    plt.savefig(path, format='svg')
    plt.close(fig)


def print_model_stats(model):
    print('Model stats =======================================================')
    for param_name, param in model.named_parameters():
        print('[print_stats] {: >30} max:{:.6} min:{:.6} mean:{:.6f} var:{:.6f}'
              .format(
                  param_name,
                  np.max(param.data.numpy()),
                  np.min(param.data.numpy()),
                  np.mean(param.data.numpy()),
                  np.var(param.data.numpy())))


def print_tensor_stats(tensor, name):
    print('{} stats ======================================================='
          .format(name))
    print('[print_tensor_stats] max:{:.6} min:{:.6} mean:{:.6f} var:{:.6f}'
          .format(
              np.max(tensor.detach().numpy()),
              np.min(tensor.detach().numpy()),
              np.mean(tensor.detach().numpy()),
              np.var(tensor.detach().numpy())))
