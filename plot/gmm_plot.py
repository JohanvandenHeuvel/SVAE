import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch

from distributions import Dirichlet, NormalInverseWishart, Gaussian

cm = plt.get_cmap("tab20")

from .plotly import save_plotly_figure, single_plotly_figure, dual_plotly_figure


def _plot_clusters(eta_theta):
    """
    Plot latent clusters of the SVAE.

    Parameters
    ----------
    eta_theta:
        parameters for clusters in latent space
    """

    def generate_ellipse(mu, Sigma):
        """
        Generate ellipse from (mu, Sigma).
        """
        t = np.linspace(0, 2 * np.pi, 100) % 2 * np.pi
        circle = np.vstack((np.sin(t), np.cos(t)))
        ellipse = 2.0 * np.dot(np.linalg.cholesky(Sigma), circle)
        return ellipse[0] + mu[0], ellipse[1] + mu[1]

    def get_component(gaussian_parameters):
        """
        Get (mu, Sigma).
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
    plots = []
    for i, (weight, (mu, Sigma)) in enumerate(zip(weights, components)):
        # don't plot clusters that are hardly visible
        if weight > 0.05:
            if isinstance(mu, torch.Tensor):
                mu = mu.cpu().detach().numpy()
            if isinstance(Sigma, torch.Tensor):
                Sigma = Sigma.cpu().detach().numpy()
            x, y = generate_ellipse(mu, Sigma)
            plots.append(
                go.Scatter(
                    x=x,
                    y=y,
                    line={"color": px.colors.qualitative.Alphabet[i], "dash": "dash"},
                    name=f"cluster_{i}",
                )
            )
    return plots


def _plot_observed_space(fig, row, col, obs=None, mu=None, classes=None):
    if obs is not None:
        x, y = zip(*obs)
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers", name="obs"), row=row, col=col
        )

    if mu is not None:
        x, y = zip(*mu)

        # set color of the data-points
        marker_color = None
        if classes is not None:
            marker_color = [px.colors.qualitative.Alphabet[i] for i in classes]

        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers", name="recon", marker={"color": marker_color}
            ),
            row=row,
            col=col,
        )


def plot_observed_space(
    obs=None,
    mu=None,
    classes=None,
    title=None,
    y_axes=None,
    x_axes=None,
):
    """

    Parameters
    ----------
    obs:
        Original observations.
    mu:
        Reconstructions of the observations.
    classes:
        Cluster assignment for the data-points.
    title:
        Title of the plot.
    y_axes:
        None to disable y-axis, else y-axis range [a, b].
    x_axes:
        None to disable x-axis, else x-axis range [a, b].
    """
    fig = single_plotly_figure(x_axes, y_axes)
    _plot_observed_space(fig, 1, 1, obs, mu, classes)
    return save_plotly_figure(fig, title)


def _plot_latent_space(fig, row, col, latent=None, eta_theta=None):
    if latent is not None:
        x, y = zip(*latent)
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers", name="latent"), row=row, col=col
        )

    if eta_theta is not None:
        plots = _plot_clusters(eta_theta)
        for i, plot in enumerate(plots):
            fig.add_trace(plot, row=row, col=col)


def plot_latent_space(
    latent=None,
    eta_theta=None,
    title=None,
    y_axes=None,
    x_axes=None,
):
    """

    Parameters
    ----------
    latent:
        Latent representation of the observations.
    eta_theta:
        Parameters for clusters in latent space.
    title:
        Title of the plot.
    y_axes:
        None to disable y-axis, else y-axis range [a, b].
    x_axes:
        None to disable x-axis, else x-axis range [a, b].
    """
    fig = single_plotly_figure(x_axes, y_axes)
    _plot_latent_space(fig, 1, 1, latent, eta_theta)
    return save_plotly_figure(fig, title)


def plot_reconstruction(
    obs=None,
    mu=None,
    latent=None,
    eta_theta=None,
    classes=None,
    title="plot",
):
    """

    Parameters
    ----------
    obs:
        Original observations.
    mu:
        Reconstructions of the observations.
    latent:
        Latent representation of the observations.
    eta_theta:
        Parameters for clusters in latent space.
    classes:
        Cluster assignment for the data-points.
    title:
        Title of the plot.
    """

    n_col = (latent is not None) + (obs is not None or mu is not None)
    assert n_col == 1 or n_col == 2

    fig = dual_plotly_figure()
    _plot_latent_space(fig, 1, 1, latent, eta_theta)
    _plot_observed_space(fig, 1, n_col, obs, mu, classes)
    return save_plotly_figure(fig, title)
