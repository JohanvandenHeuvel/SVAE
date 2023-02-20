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
  #     bad VAE       #
  #                   #
  #####################
          (b)
  ---------------------
  which plots VAE without
  residual connections
  ---------------------
 
  #####################
  #                   #
  #   bad global      #
  #                   #
  #####################
          (c)
  ---------------------
  which plots bad cluster
  initialization
  ---------------------
"""

import torch
import wandb

from distributions import Gaussian
from matrix_ops import unpack_dense
from plot.gmm_plot import plot_latent_space
from run_gmm import get_data, get_network
from run_gmm import hyperparameters as gmm_hyperparameters
from svae.gmm import SVAE
from svae.gmm.global_optimization import initialize_global_gmm_parameters
from svae.gmm.local_optimization import local_optimization
from vae import VAE


def generate_good_init():
    """
    Generate the good initialization plot. This is using the initial conditions used in the main code.

    Returns
    -------

    """
    print(f"Generating SVAE plot...")

    # get data
    observations = get_data()

    # get network
    network = get_network()

    # get global parameters
    K = 15
    _, D = observations.shape
    eta_theta = initialize_global_gmm_parameters(
        K, D, alpha=1.0, niw_conc=1.0, random_scale=3.0
    )
    # get SVAE model
    model = SVAE(network)

    # get the latents
    data = torch.tensor(observations).to(model.vae.device).double()
    potentials = model.encode(data)
    x, eta_x, label_stats, _, _ = local_optimization(potentials, eta_theta)

    # get encoded means
    gaussian_stats = Gaussian(eta_x).expected_stats()
    _, Ex, _, _ = unpack_dense(gaussian_stats)

    plot_latent_space(
        latent=Ex.cpu().detach().numpy(),
        eta_theta=eta_theta,
        title="good_init",
        x_axes=[-8, 8],
        y_axes=[-8, 8],
    )


def generate_bad_vae():
    """
    Generate the bad VAE plot. This is not using residual connections.

    Returns
    -------

    """
    print(f"Generating SVAE plot...")

    # get data
    observations = get_data()

    # get network
    network = VAE(**gmm_hyperparameters["VAE_parameters"])

    # get global parameters
    K = 15
    _, D = observations.shape
    eta_theta = initialize_global_gmm_parameters(
        K, D, alpha=1.0, niw_conc=1.0, random_scale=3.0
    )

    # get SVAE model
    model = SVAE(network)

    # get latents
    data = torch.tensor(observations).to(model.vae.device).double()
    potentials = model.encode(data)
    x, eta_x, label_stats, _, _ = local_optimization(potentials, eta_theta)

    # get encoded means
    gaussian_stats = Gaussian(eta_x).expected_stats()
    _, Ex, _, _ = unpack_dense(gaussian_stats)

    plot_latent_space(
        latent=Ex.cpu().detach().numpy(),
        eta_theta=eta_theta,
        title="bad_vae",
        x_axes=[-8, 8],
        y_axes=[-8, 8],
    )


def generate_bad_global():
    """
    Generate the bad global plot. This is using bad priors for the global parameters.

    Returns
    -------

    """
    print(f"Generating SVAE plot...")

    # get data
    observations = get_data()

    # get network
    network = get_network()

    # get global parameters
    K = 15
    _, D = observations.shape
    eta_theta = initialize_global_gmm_parameters(
        K, D, alpha=1.0, niw_conc=1.0, random_scale=0.75
    )

    # get SVAE model
    model = SVAE(network)

    # get latents
    data = torch.tensor(observations).to(model.vae.device).double()
    potentials = model.encode(data)
    x, eta_x, label_stats, _, _ = local_optimization(potentials, eta_theta)

    # get encoded means
    gaussian_stats = Gaussian(eta_x).expected_stats()
    _, Ex, _, _ = unpack_dense(gaussian_stats)

    plot_latent_space(
        latent=Ex.cpu().detach().numpy(),
        eta_theta=eta_theta,
        title="bad_global",
        x_axes=[-8, 8],
        y_axes=[-8, 8],
    )


if __name__ == "__main__":
    wandb.init(mode="disabled")
    generate_good_init()
    generate_bad_vae()
    generate_bad_global()
