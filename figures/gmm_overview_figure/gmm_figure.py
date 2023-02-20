"""
Generate the following figures:

  #####################
  #                   #
  #   observations    #
  #                   #
  #####################
          (a)
  ---------------------
  which just plots the
  normal observations
  without anything else
  ---------------------

  #####################
  #                   #
  #     GMM plot      #
  #                   #
  #####################
          (b)
  ---------------------
  which plots the
  clusters found by the
  GMM
  ---------------------

  #####################
  #                   #
  #       VAE         #
  #                   #
  #####################
          (c)
  ---------------------
  which plots the
  density estimation
  from the VAE
  ---------------------

  #####################
  #                   #
  #       SVAE        #
  #                   #
  #####################
          (d)
  ---------------------
  which plots the
  density estimation
  and the classes from
  the SVAE
  ---------------------

"""

import time

import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import torch
import wandb
from plotly.subplots import make_subplots

from GMM import GMM
from plot.gmm_plot import plot_observed_space
from run_gmm import get_data, get_network
from run_gmm import hyperparameters as gmm_hyperparameters
from svae.gmm import SVAE
from vae.run_vae import hyperparameters as vae_hyperparameters


def generate_observations():
    print(f"Generating observations plot...")
    observations = get_data()

    fig = make_subplots(rows=1, cols=1)
    fig.update_layout(
        showlegend=False,
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
    )
    fig.update_yaxes(title="y", visible=False, showticklabels=False)
    fig.update_xaxes(title="x", visible=False, showticklabels=False)

    x, y = zip(*observations)
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="obs"), row=1, col=1)

    # save to file
    title = "observations"

    # see https://github.com/plotly/plotly.py/issues/3469
    plotly.io.write_image(fig, f"{title}.pdf", format="pdf")
    time.sleep(2)
    plotly.io.write_image(fig, f"{title}.pdf", format="pdf")
    print(f"save plot to {title}.pdf")


def generate_gmm():
    print(f"Generating GMM plot...")

    def _plot_clusters():
        def generate_ellipse(mu, Sigma):
            """
            Generate ellipse from (mu, Sigma).
            """
            t = np.linspace(0, 2 * np.pi, 100) % 2 * np.pi
            circle = np.vstack((np.sin(t), np.cos(t)))
            ellipse = 2.0 * np.dot(np.linalg.cholesky(Sigma), circle)
            return ellipse[0] + mu[0], ellipse[1] + mu[1]

        """
        plot latent clusters
        """
        plots = []
        n = len(gmm.mu)
        for i in range(n):
            x, y = generate_ellipse(gmm.mu[i], gmm.sigma[i])
            plots.append(
                go.Scatter(
                    x=x,
                    y=y,
                    line={"color": px.colors.qualitative.Alphabet[i], "dash": "dash"},
                    name=f"cluster_{i}",
                )
            )
        return plots

    def plot_reconstruction(title):
        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(
            showlegend=False,
            autosize=False,
            width=800,
            height=800,
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
        )

        fig.add_trace(
            go.Scatter(
                x=observations[:, 0], y=observations[:, 1], mode="markers", name="obs"
            ),
            row=1,
            col=1,
        )

        n = len(gmm.mu)
        for i in range(n):
            fig.add_trace(
                go.Scatter(
                    x=[gmm.mu[i, 0]],
                    y=[gmm.mu[i, 1]],
                    mode="markers",
                    name="recon",
                    marker={"color": [px.colors.qualitative.Alphabet[i]]},
                ),
                row=1,
                col=1,
            )
        plots = _plot_clusters()
        for i, plot in enumerate(plots):
            fig.add_trace(plot, row=1, col=1)

        fig.update_yaxes(title="y", visible=False, showticklabels=False)
        fig.update_xaxes(title="x", visible=False, showticklabels=False)
        plotly.io.write_image(fig, "gmm.pdf", format="pdf")
        return fig

    n_clusters = 15
    dim = 2
    # Create a Gaussian Mixture Model
    gmm = GMM(n_clusters, dim)
    # Training the GMM using EM
    observations = get_data()

    # Initialize EM algorithm with data
    gmm.init_em(observations)
    num_iters = 30

    # plotting
    for e in range(num_iters):
        # E-step
        gmm.e_step()
        # M-step
        gmm.m_step()
        # plotting
        plot_reconstruction(title="Iteration: " + str(e + 1))
    print(f"save plot to gmm.pdf")


def generate_vae():
    print(f"Generating VAE plot...")
    # get data
    observations = get_data()
    # get network
    network = get_network()
    # train network
    network.fit(
        observations, **vae_hyperparameters["VAE_train_parameters"], verbose=False
    )
    # get predictions
    mu_x, log_var_x, mu_z, log_var_z = network.forward(
        torch.tensor(observations).double()
    )

    plot_observed_space(
        obs=observations,
        mu=mu_x.detach().numpy(),
        title="VAE",
    )


def generate_svae():
    print(f"Generating SVAE plot...")
    # get data
    observations = get_data()
    # get network
    network = get_network()

    # get SVAE model
    model = SVAE(network)
    # train model
    model.fit(observations, **gmm_hyperparameters["SVAE_parameters"], verbose=False)
    # get predictions
    mu_y, log_var_y, _, classes = model.forward(torch.tensor(observations).double())

    plot_observed_space(
        obs=observations,
        mu=mu_y.squeeze().detach().numpy(),
        classes=classes,
        title="SVAE",
    )


if __name__ == "__main__":

    wandb.init(mode="disabled")
    generate_observations()
    try:
        generate_gmm()
    except np.linalg.LinAlgError:
        # sometimes the covariance matrix is not invertible
        print("GMM failed")
        pass
    generate_vae()
    generate_svae()
