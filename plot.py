import matplotlib.pyplot as plt
import numpy as np
import torch

from dense import unpack_dense

import os

from distributions import Dirichlet, NormalInverseWishart


def plot_latent(latents, eta_theta, K, title=None, save_path=None):
    """
    Plot latent dimension, including clusters, of the SVAE
    """

    fig, ax = plt.subplots()
    _plot_latent(ax, latents, eta_theta, K)

    if save_path is not None:
        if title is None:
            raise ValueError(f"saving requires title but title is {title}")
        fig.savefig(os.path.join(save_path, title))
    else:
        plt.plot()


def _plot_latent(ax, latents, eta_theta, K=15, title=None):
    """
    Plot latent dimension, including clusters, of the SVAE
    """

    def get_component(niw_param):
        """
        Get (mu, Sigma)
        """
        neghalfJ, h, _, _ = unpack_dense(niw_param)
        J = -2 * neghalfJ
        return torch.linalg.solve(J, h), torch.inverse(J)

    def normalize(arr):
        """
        Make sure that 'arr' sums to 1.0
        """
        arr = arr.detach().numpy()
        return np.minimum(1.0, arr / np.sum(arr) * K)

    """
    Get objects for plotting
    """
    dir_param, niw_param = eta_theta
    weights = normalize(torch.exp(Dirichlet(dir_param).expected_stats()))
    components = map(get_component, NormalInverseWishart(niw_param).expected_stats())

    """
    plot latent data-points
    """
    x, y = zip(*latents.detach().numpy())
    ax.scatter(x, y)

    """
    plot latent clusters
    """
    for weight, (mu, Sigma) in zip(weights, components):
        x, y = generate_ellipse(mu.detach().numpy(), Sigma.detach().numpy())
        ax.plot(x, y, alpha=weight, linestyle="-", linewidth=3)

    ax.set_title(title)


def plot_scatter(data, title=None, save_path=None):
    """
    Make scatter plot for data of the form [(x1, y1), ..., (xi, yi), ...]
    """
    fig, ax = plt.subplots()
    _plot_scatter(ax, data, title)

    if save_path is not None:
        if title is None:
            raise ValueError(f"saving requires title but title is {title}")
        fig.savefig(os.path.join(save_path, title))
        plt.close(fig)
    else:
        plt.plot()


def _plot_scatter(ax, data, title=None):
    """
    Make scatter plot for data of the form [(x1, y1), ..., (xi, yi), ...]
    """
    x, y = zip(*data)
    ax.scatter(x, y)
    ax.set_title(title)

def plot_loss(loss, title=None, save_path=None):

    fig, ax = plt.subplots()
    ax.plot(loss)
    ax.set_title(title)
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")

    if save_path is not None:
        if title is None:
            raise ValueError(f"saving requires title but title is {title}")
        fig.savefig(os.path.join(save_path, title))
        plt.close(fig)
    else:
        plt.plot()


def generate_ellipse(mu, Sigma):
    """
    Generate ellipse from a (mu, Sigma)
    """
    t = np.hstack([np.arange(0, 2 * np.pi, 0.01), 0])
    circle = np.vstack([np.sin(t), np.cos(t)])
    ellipse = 2.0 * np.dot(np.linalg.cholesky(Sigma), circle)
    return ellipse[0] + mu[0], ellipse[1] + mu[1]
