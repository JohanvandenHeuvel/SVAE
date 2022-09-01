import os

import matplotlib.pyplot as plt
import numpy as np
import torch.linalg

from matrix_ops import unpack_dense


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
    n_samples, sample_length, n_latents = latents.shape

    fig, axs = plt.subplots(n_latents, 1, figsize=(10, n_latents * 4))
    for dim_i in range(n_latents):
        latent_i = latents[..., dim_i]
        ax_i = axs[dim_i]

        # plot some samples
        for sample_j in range(5):
            ax_i.plot(latent_i[sample_j], "--", alpha=0.4)

        # plot mean
        ax_i.plot(latent_i.mean(0), linewidth=2)

        # plot vertical line for prefix
        ax_i.plot(
            [prefix - 0.5, prefix - 0.5],
            [ax_i.get_ylim()[0], ax_i.get_ylim()[1]],
            "--",
            color="r",
            linewidth=2,
        )

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


def plot_parameters(A, Q, Sigma, mu, title=None, save_path=None):
    fig, axs = plt.subplots(2, 2)

    cmap = "coolwarm"
    im_A = axs[0, 0].matshow(A.cpu().detach().numpy(), cmap=cmap)
    im_Q = axs[0, 1].matshow(Q.cpu().detach().numpy(), cmap=cmap)
    im_Sigma = axs[1, 0].matshow(Sigma.cpu().detach().numpy(), cmap=cmap)
    im_mu = axs[1, 1].matshow(mu[..., None].cpu().detach().numpy(), cmap=cmap)

    fig.colorbar(im_A, ax=axs[0, 0])
    fig.colorbar(im_Q, ax=axs[0, 1])
    fig.colorbar(im_Sigma, ax=axs[1, 0])
    fig.colorbar(im_mu, ax=axs[1, 1])

    axs[0, 0].axis("off")
    axs[0, 1].axis("off")
    axs[1, 0].axis("off")
    axs[1, 1].axis("off")

    axs[0, 0].title.set_text("A")
    axs[0, 1].title.set_text("Q")
    axs[1, 0].title.set_text("Sigma")
    axs[1, 1].title.set_text("mu")

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


def plot_info_parameters(J11, J12, J22, J21, title=None, save_path=None):
    fig, axs = plt.subplots(2, 2)

    cmap = "coolwarm"
    im_J11 = axs[0, 0].matshow(J11.cpu().detach().numpy(), cmap=cmap)
    im_J12 = axs[0, 1].matshow(J12.cpu().detach().numpy(), cmap=cmap)
    im_J22 = axs[1, 0].matshow(J22.cpu().detach().numpy(), cmap=cmap)
    im_J21 = axs[1, 1].matshow(J21.cpu().detach().numpy(), cmap=cmap)

    fig.colorbar(im_J11, ax=axs[0, 0])
    fig.colorbar(im_J12, ax=axs[0, 1])
    fig.colorbar(im_J22, ax=axs[1, 0])
    fig.colorbar(im_J21, ax=axs[1, 1])

    axs[0, 0].axis("off")
    axs[0, 1].axis("off")
    axs[1, 0].axis("off")
    axs[1, 1].axis("off")

    axs[0, 0].title.set_text("J11")
    axs[0, 1].title.set_text("J12")
    axs[1, 0].title.set_text("J22")
    axs[1, 1].title.set_text("J21")

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


def plot_potentials(potentials, prefix, title=None, save_path=None):

    J, h, _, _ = unpack_dense(potentials)
    J = torch.linalg.diagonal(J)

    J = J[:prefix].cpu().detach().numpy()
    h = h[:prefix].cpu().detach().numpy()

    x = np.linspace(0, prefix, prefix)

    _, n_latents = J.shape

    fig, axs = plt.subplots(n_latents, 1, figsize=(10, n_latents * 4))
    for dim_i in range(n_latents):
        ax_i = axs[dim_i]
        ax_i.plot(h[:, dim_i])
        ax_i.fill_between(
            x, h[:, dim_i] - J[:, dim_i], h[:, dim_i] + J[:, dim_i], alpha=0.1
        )

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
