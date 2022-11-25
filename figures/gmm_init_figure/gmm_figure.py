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
from plot.gmm_plot import plot_reconstruction
from run_gmm import get_data, get_network
from run_gmm import hyperparameters as gmm_hyperparameters
from svae.gmm import SVAE
from svae.gmm.global_optimization import initialize_global_gmm_parameters
from svae.gmm.local_optimization import local_optimization
from vae import VAE


def generate_observations():
    pass


def generate_bad_vae():
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

    # DO STUFF
    data = torch.tensor(observations).to(model.vae.device).double()
    potentials = model.encode(data)

    x, eta_x, label_stats, _, _ = local_optimization(potentials, eta_theta)

    # get encoded means
    gaussian_stats = Gaussian(eta_x).expected_stats()
    _, Ex, _, _ = unpack_dense(gaussian_stats)

    plot_reconstruction(
        latent=Ex.cpu().detach().numpy(),
        eta_theta=eta_theta,
        title="bad_vae",
    )


def generate_bad_global():
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

    # DO STUFF
    data = torch.tensor(observations).to(model.vae.device).double()
    potentials = model.encode(data)

    x, eta_x, label_stats, _, _ = local_optimization(potentials, eta_theta)

    # get encoded means
    gaussian_stats = Gaussian(eta_x).expected_stats()
    _, Ex, _, _ = unpack_dense(gaussian_stats)

    plot_reconstruction(
        latent=Ex.cpu().detach().numpy(),
        eta_theta=eta_theta,
        title="bad_global",
    )


if __name__ == "__main__":
    wandb.init(mode="disabled")
    # generate_observations()
    generate_bad_vae()
    generate_bad_global()
