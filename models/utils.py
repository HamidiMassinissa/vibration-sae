import os
import numpy as np
from deprecated import deprecated
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import sklearn.metrics

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


def plot_loss(loss, title=None):
    print('[plot_loss]\ntitle={}\nlosses.shape={}'.format(title, loss.shape))
    fig = plt.figure(figsize=(20, 5))
    plt.plot(np.concatenate(loss))
    # save_figure(fig, title)
    return fig


def plot_prediction(x, x_hat, enc, title):
    B = x.shape[0]
    fig = plt.figure(figsize=(15, 5*B))
    for i, (x_i, x_hat_i, enc_i) in enumerate(zip(x, x_hat, enc)):
        # plt.subplot(B, 1, 2*i+1)
        plt.subplot(B, 2, 2*i+1)
        plt.plot(x_i.detach().numpy(), color='b')
        # plt.plot(x_norm_i.detach().numpy(), color='g')
        plt.plot(x_hat_i.detach().numpy(), color='r')
        print('I: [plot_prediction] x_i = %s' % np.array2string(x_i.detach().numpy(), threshold=np.inf).replace('\n', ''))
        print('I: [plot_prediction] x_i.detach().numpy().shape = {}'.format(x_i.detach().numpy().shape))
        print('I: [plot_prediction] x_hat_i = %s' % np.array2string(x_hat_i.detach().numpy(), threshold=np.inf).replace('\n', ''))
        # plt.subplot(B, 1, 2*i+2)
        print('I: [plot_prediction] sklearn.mse(xi, xhat) = {}'.format(sklearn.metrics.mean_squared_error(x_i.detach().numpy(), x_hat_i.detach().numpy())))
        plt.subplot(B, 2, 2*i+2)
        plt.plot(enc_i.detach().numpy(), color='g')
    save_figure(fig, 'recons-{}-batch-{}-epoch-{}'.format(title, config.batch_idx, config.epoch_idx))


def plot_encoding(enc):
    B = enc.shape[0]
    fig = plt.figure(figsize=(6, 20))
    for i, enc_i in enumerate(enc):
        plt.subplot(B, 1, i+1)
        plt.plot(enc_i.detach().numpy(), color='g')
    save_figure(fig, 'encoding-batch-{}'.format(config.batch_idx))


def plot_hidden(h):
    fig = plt.figure()
    h = np.sort(h[0].detach().numpy())
    pdf = scipy.stats.norm.pdf(h, np.mean(h), np.std(h))
    plt.plot(h, pdf)
    save_figure(fig, 'hidden values distribution')


def plot_tSNE(encoding):
    print('[plot_tSNE] encoding.shape = {}'.format(encoding.shape))
    fig = plt.figure()
    palette = np.array(sns.color_palette("hls", encoding.shape[0]))
    embedded = TSNE(n_components=2, random_state=config.SEED).fit_transform(encoding)
    plt.scatter(embedded[:, 0], embedded[:, 1], c=palette)
    save_figure(fig, 'tSNE')
