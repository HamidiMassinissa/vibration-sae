import numpy as np
from numpy import linalg
import sklearn
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

from .utils import save_figure
from config import Configuration as config
matplotlib.use('Agg')


def plot_tSNE(encoding):
    print('[plot_tSNE] encoding.shape = {}'.format(encoding.shape))
    fig = plt.figure()
    palette = np.array(sns.color_palette("hls", encoding.shape[0]))
    embedded = TSNE(
        n_components=2, random_state=config.SEED
    ).fit_transform(encoding)
    plt.scatter(embedded[:, 0], embedded[:, 1], c=palette)
    save_figure(fig, 'tSNE')


def scatter(x):
    print('[scatter] x.shape = {}'.format(x.shape))
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", x.shape[0]))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    return f, ax, sc


def record_tSNE_gradient_descent(encodings):
    # This list will contain the positions of the map points at every iteration.
    positions = []

    def _gradient_descent(objective, p0, it, n_iter, n_iter_without_progress=30,
                          momentum=0.5, learning_rate=1000.0, min_gain=0.01,
                          min_grad_norm=1e-7, min_error_diff=1e-7, verbose=0,
                          n_iter_check=1, kwargs=None, args=[]):
        # Source https://github.com/oreillymedia/t-SNE-tutorial
        # The documentation of this function can be found in scikit-learn's code
        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(np.float).max
        best_error = np.finfo(np.float).max
        best_iter = 0

        for i in range(it, n_iter):
            # We save the current position.
            positions.append(p.copy())

            new_error, grad = objective(p, *args)
            error_diff = np.abs(new_error - error)
            error = new_error
            grad_norm = linalg.norm(grad)

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                break
            if min_grad_norm >= grad_norm:
                break
            if min_error_diff >= error_diff:
                break

            inc = update * grad >= 0.0
            dec = np.invert(inc)
            gains[inc] += 0.05
            gains[dec] *= 0.95
            np.clip(gains, min_gain, np.inf)
            grad *= gains
            update = momentum * update - learning_rate * grad
            p += update

        return p, error, i
    sklearn.manifold.t_sne._gradient_descent = _gradient_descent

    X_proj = TSNE(
        n_components=2, random_state=config.SEED
    ).fit_transform(encodings)

    X_iter = np.dstack(position.reshape(-1, 2)
                       for position in positions)
    f, ax, sc = scatter(X_iter[..., -1])

    def make_frame_mpl(t):
        i = int(t*40)
        x = X_iter[..., i]
        sc.set_offsets(x)
        return mplfig_to_npimage(f)

    animation = mpy.VideoClip(make_frame_mpl,
                              duration=X_iter.shape[2]/40.)
    animation.write_gif("./animation.gif", fps=20)
