import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import torch

import os


def plot_observations(ax, obs, prefix):
    """Plot 1d frames over time.

    Parameters
    ----------
    ax:
        ax on which to plot.
    obs: tensor
        Image matrix.
    prefix: int
        Point at which to draw vertical line.
    """
    ax.matshow(obs, cmap="gray")
    ax.plot([prefix - 0.5, prefix - 0.5], [-0.5, len(obs)], "r", linewidth=2)
    ax.axis("off")


def plot_latents(ax, latents, mean, variance):
    """Plot the latent states.

    Parameters
    ----------
    ax:
        ax on which to plot.
    latents:
        Sampled latent states, from Normal(mean, variance).
    mean:
        Mean of latent states.
    variance:
        Variance of latent states.
    """
    colors = cm.rainbow(np.linspace(0, 1, len(latents)))
    x = np.linspace(0, 100, 100)
    for j in range(len(latents)):
        ax.plot(latents[j], "--", c=colors[j], alpha=0.8)
        ax.plot(mean[j], c=colors[j], alpha=0.8)
        ax.fill_between(x, mean[j] - variance[j], mean[j] + variance[j], color=colors[j], alpha=0.1)


def plot(obs, samples, latent_samples, latent_means, latent_vars, prefix=25, title=None, save_path=None):
    """

    Parameters
    ----------
    obs:
        observations
    samples:
        reconstructed observations
    latent_samples:
        sampled latent states
    latent_means:
        mean for latent states
    latent_vars:
        variance for latent states
    prefix:
        after which time to zero-out the data
    title: String
        title for the plot
    save_path: String
        where to save the plot
    """
    n_samples = len(samples)
    fig, axs = plt.subplots(3, n_samples, figsize=(n_samples * 10, 10))
    for i in range(n_samples):
        ax = axs[:, i]

        plot_observations(ax[0], obs.T, prefix)
        plot_observations(ax[1], samples[i].T, prefix)

        variance = np.diagonal(
            latent_vars[i], offset=0, axis1=-2, axis2=-1
        )  # de-diagonalize
        plot_latents(ax[2], latent_samples[i].T, latent_means[i].T, variance.T)

    fig.suptitle(title)
    fig.tight_layout()
    # save the figure to disk or show it
    if save_path is not None:
        if title is None:
            raise ValueError(f"saving requires title but title is {title}")
        fig.savefig(os.path.join(save_path, title))
        plt.close(fig)
    else:
        plt.plot()
