import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from distributions import Dirichlet, NormalInverseWishart, Gaussian


def plot_reconstruction(
    data, recon, latent, eta_theta=None, title=None, save_path=None
):
    """

    Parameters
    ----------
    data:
        observations
    recon:
        reconstructions of the observations
    latent:
        latent representation of the observations
    eta_theta:
        parameters for clusters in latent space
    title: String
        title for the plot
    save_path: String
        where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # plot the latent dimension in the left plot
    _plot_scatter(ax1, latent, title="latents")
    if eta_theta is not None:
        _plot_clusters(ax1, eta_theta)
    # plot the observations in the right plot
    _plot_scatter(ax2, data)
    _plot_scatter(ax2, recon, title="reconstruction")

    # save the figure to disk or show it
    if save_path is not None:
        if title is None:
            raise ValueError(f"saving requires title but title is {title}")
        fig.savefig(os.path.join(save_path, title))
        plt.close(fig)
    else:
        plt.plot()


def plot_loss(loss, title=None, save_path=None):
    """

    Parameters
    ----------
    loss: List
        loss values for every epoch
    title: String
        title for the plot
    save_path: String
        where to save the plot

    Returns
    -------

    """

    if loss == 0:
        # TODO needed to handle model loading without losses saved, not most elegant solution
        return

    recon_loss, kld_loss = list(zip(*loss))
    fig, ax = plt.subplots()
    ax.plot(recon_loss)
    ax.plot(kld_loss)
    ax.set_title(title)
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.legend(["recon", "kld"])

    # save the figure to disk or show it
    if save_path is not None:
        if title is None:
            raise ValueError(f"saving requires title but title is {title}")
        fig.savefig(os.path.join(save_path, title))
        plt.close(fig)
    else:
        plt.plot()


def _plot_clusters(ax, eta_theta, title=None):
    """
    Plot latent clusters of the SVAE

    Parameters
    ----------
    ax:
        which ax to plot on
    eta_theta:
        parameters for clusters in latent space
    title: String
        title for the plot

    """

    def generate_ellipse(mu, Sigma):
        """
        Generate ellipse from a (mu, Sigma)
        """
        # t = np.hstack([np.arange(0, 2 * np.pi, 0.01), 0])
        t = np.linspace(0, 2 * np.pi, 100) % 2 * np.pi
        circle = np.vstack((np.sin(t), np.cos(t)))
        ellipse = 2.0 * np.dot(np.linalg.cholesky(Sigma), circle)
        return ellipse[0] + mu[0], ellipse[1] + mu[1]

    def get_component(gaussian_parameters):
        """
        Get (mu, Sigma)
        """
        loc, scale = Gaussian(gaussian_parameters.unsqueeze(0)).natural_to_standard()

        return loc.squeeze(), scale.squeeze()

    def normalize(arr):
        """
        Make sure that 'arr' sums to 1.0
        """
        return np.minimum(1.0, arr / np.sum(arr) * K)

    """
    Get objects for plotting
    """
    dir_param, niw_param = eta_theta
    K = len(dir_param)
    weights = normalize(
        torch.exp(Dirichlet(dir_param).expected_stats()).cpu().detach().numpy()
    )
    components = map(get_component, NormalInverseWishart(niw_param).expected_stats())

    """
    plot latent clusters
    """
    for weight, (mu, Sigma) in zip(weights, components):
        x, y = generate_ellipse(mu.cpu().detach().numpy(), Sigma.cpu().detach().numpy())
        ax.plot(x, y, alpha=weight, linestyle="-", linewidth=3)

    ax.set_title(title)


def _plot_scatter(ax, data, title=None):
    """
    Make scatter plot for data of the form [(x1, y1), ..., (xi, yi), ...]
    """
    x, y = zip(*data)
    ax.scatter(x, y)
    ax.set_title(title)
