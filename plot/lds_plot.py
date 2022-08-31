import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_observations(obs, samples, variance, title="plot", save_path=None):
    N = obs.shape[-1]
    fig, axs = plt.subplots(N, 1, figsize=(10, N * 4))

    x = np.linspace(0, 100, 100)
    for n in range(N):
        ax = axs[n]
        ax.plot(obs[:, n], label="observed", alpha=0.8)
        ax.plot(
            samples[:, n], linestyle="dashed", label="sampled", alpha=0.8,
        )
        ax.fill_between(
            x, obs[:, n] - variance[:, n], obs[:, n] + variance[:, n], alpha=0.1
        )
        ax.legend()
        ax.set_xlabel("time")
        ax.set_ylabel("obs y")

    fig.suptitle(title)
    fig.tight_layout()
    # save the figure to disk or show it
    if save_path is not None:
        if title is None:
            raise ValueError(f"saving requires title but title is {title}")
        fig.savefig(os.path.join(save_path, title))
        plt.close(fig)
    else:
        plt.show()


def plot_video_observations(ax, obs, prefix):
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


def plot_latents(latents, prefix, title=None, save_path=None):
    """Plot the latent states.

    Parameters
    ----------
    latents:
        Sampled latent states, from Normal(mean, variance).
    """
    N = len(latents)
    fig, axs = plt.subplots(N, 1, figsize=(10, N * 4))
    for j in range(N):
        axs[j].plot(latents[j])
        axs[j].plot([prefix - 0.5, prefix - 0.5], [latents[j].min(), latents[j].max()], "--", "r", linewidth=2)


    # save the figure to disk or show it
    if save_path is not None:
        if title is None:
            raise ValueError(f"saving requires title but title is {title}")
        fig.savefig(os.path.join(save_path, title))
        plt.close(fig)
    else:
        plt.show()


def plot(
    obs, samples, prefix=25, title=None, save_path=None,
):
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
    fig, axs = plt.subplots(1, 1, figsize=(20, 10))
    mean_image = samples.mean(0)
    sample_images = np.hstack(samples[:5])
    big_image = np.hstack((obs, mean_image, sample_images))
    plot_video_observations(axs, big_image.T, prefix)

    fig.suptitle(title)
    fig.tight_layout()
    # save the figure to disk or show it
    if save_path is not None:
        if title is None:
            raise ValueError(f"saving requires title but title is {title}")
        fig.savefig(os.path.join(save_path, title))
        plt.close(fig)
    else:
        plt.show()
