import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from distributions import Dirichlet, NormalInverseWishart, Gaussian

import time

cm = plt.get_cmap("tab20")


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


def plot_reconstruction(
    obs=None,
    mu=None,
    latent=None,
    eta_theta=None,
    classes=None,
    title="",
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

    n_col = (latent is not None) + (obs is not None)
    assert n_col == 1 or n_col == 2

    fig = make_subplots(rows=1, cols=n_col, subplot_titles=("Latent", "Observation") if n_col == 2 else None)
    fig.update_layout(
        showlegend=False,
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
    )

    """
    plot the latent dimension in the left plot
    """
    if latent is not None:
        x, y = zip(*latent)
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="latent"), row=1, col=1)

    if eta_theta is not None:
        plots = _plot_clusters(eta_theta)
        for i, plot in enumerate(plots):
            fig.add_trace(plot, row=1, col=1)

    """
    plot the observations in the right plot
    """
    if obs is not None:
        x, y = zip(*obs)
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="obs"), row=1, col=n_col)

    if mu is not None:
        x, y = zip(*mu)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="recon",
                marker={
                    "color": [px.colors.qualitative.Alphabet[i] for i in classes] if classes is not None else None
                },
            ),
            row=1,
            col=n_col,
        )

    # remove axes
    fig.update_yaxes(title="y", visible=False, showticklabels=False)
    fig.update_xaxes(title="x", visible=False, showticklabels=False)

    # save to file
    # see https://github.com/plotly/plotly.py/issues/3469
    plotly.io.write_image(fig, f"{title}.pdf", format="pdf")
    time.sleep(2)
    plotly.io.write_image(fig, f"{title}.pdf", format="pdf")
    print(f"save plot to {title}.pdf")
    return fig
