import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch.linalg

from distributions import MatrixNormalInverseWishart
from matrix_ops import unpack_dense


def plot_list(l):
    x = np.arange(0, len(l))
    fig, ax = plt.subplots()
    ax.bar(x, l)
    return fig


def plot_observations(obs, title=None, figsize=(10, 10), xrange=None):
    """Plot the observations."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(obs.T, cmap="gray")
    ax.axis("tight")
    if xrange is not None:
        ax.set_xlim(xrange)
    # ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_latents(latents, prefix=None, title=None, figsize=(10, 10), xrange=None):
    n_samples, sample_length, n_latents = latents.shape

    fig, axs = plt.subplots(1, 1, figsize=figsize)
    for dim_i in range(n_latents):
        latent_i = latents[..., dim_i]
        ax_i = axs

        # plot mean
        ax_i.plot(latent_i.mean(0), linewidth=2, alpha=0.8)

        # plot vertical line for prefix
        if prefix is not None:
            ax_i.plot(
                [prefix - 0.5, prefix - 0.5],
                [ax_i.get_ylim()[0], ax_i.get_ylim()[1]],
                "--",
                color="r",
                linewidth=2,
            )

    if xrange is not None:
        axs.set_xlim(xrange)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_potentials(potentials, prefix, title=None, figsize=(10, 10)):
    J, h, _, _ = unpack_dense(potentials)
    J = torch.linalg.diagonal(J)

    J = J[:prefix].cpu().detach().numpy()
    h = h[:prefix].cpu().detach().numpy()

    _, n_latents = J.shape
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    for dim_i in range(n_latents):
        ax_i = axs
        ax_i.plot(h[:, dim_i])

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot(
    obs,
    samples,
    prefix=25,
    title=None,
):
    def plot_video(ax, obs, prefix):
        ax.matshow(obs, cmap="gray", aspect="auto")
        ax.plot([prefix - 0.5, prefix - 0.5], [-0.5, len(obs)], "r", linewidth=2)
        ax.axis("off")

    fig, axs = plt.subplots(1, 1, figsize=(30, 10))
    mean_image = samples.mean(0)
    sample_images = np.hstack(samples[:5])
    # big_image = np.hstack((obs, mean_image, sample_images))
    big_image = sample_images
    # big_image = np.hstack((obs, mean_image))
    plot_video(axs, big_image.T, prefix)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_info_parameters(J11, J12, J22, J21, A, Q, title=None, figsize=(10, 10)):
    cmap = "coolwarm"
    fig, axs = plt.subplots(2, 3, figsize=figsize)

    def _plot(i, j, mat, title):
        im = axs[i, j].matshow(mat.cpu().detach().numpy(), cmap=cmap)
        fig.colorbar(im, ax=axs[i, j])
        axs[i, j].axis("off")
        axs[i, j].title.set_text(title)

    _plot(0, 0, J11, "J11")
    _plot(0, 1, J12, "J12")
    _plot(1, 0, J22, "J22")
    _plot(1, 1, J21, "J21")
    _plot(0, 2, A, "A")
    _plot(1, 2, Q, "Q")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_global(mniw_param, title=None, figsize=(10, 10)):
    A, B, C, d = mniw_param

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    def _plot(i, j, mat, title):
        im = axs[i, j].matshow(mat.cpu().detach().numpy())
        fig.colorbar(im, ax=axs[i, j])
        axs[i, j].title.set_text(title)

    _plot(0, 0, A, "A")
    _plot(0, 1, B, "B")
    _plot(1, 0, C, "C")
    _plot(1, 1, d[None, :], "d")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_standard_params_Sigma(mniw_param):
    K, M, Phi, nu = MatrixNormalInverseWishart(mniw_param).natural_to_standard()

    # get vectorized form
    Sigma = torch.kron(K.contiguous(), Phi)

    return px.imshow(Sigma.cpu().detach().numpy())


def plot_standard_params_mu(mniw_param):
    K, M, Phi, nu = MatrixNormalInverseWishart(mniw_param).natural_to_standard()

    return px.imshow(M.cpu().detach().numpy())
