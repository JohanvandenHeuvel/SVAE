import os
import pathlib
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dense import pack_dense, unpack_dense
from distributions import (
    Gaussian,
    NormalInverseWishart,
    Dirichlet,
    exponential_kld,
)
from plot.plot import plot_reconstruction
from svae.local_optimization import local_optimization
from svae.global_optimization import natural_gradient, initialize_global_parameters
from vae import VAE


def prior_kld(
    eta_theta: Tuple[torch.Tensor, torch.Tensor],
    eta_theta_prior: Tuple[torch.Tensor, torch.Tensor],
) -> float:
    dir_params, niw_params = eta_theta
    dir_params_prior, niw_params_prior = eta_theta_prior

    dir = Dirichlet(dir_params)
    dir_prior = Dirichlet(dir_params_prior)
    dir_kld = exponential_kld(dir, dir_prior)

    niw = NormalInverseWishart(niw_params)
    niw_prior = NormalInverseWishart(niw_params_prior)
    niw_kld = exponential_kld(niw, niw_prior)

    return dir_kld + niw_kld


class SVAE:
    def __init__(self, vae: VAE):
        self.vae = vae
        self.device = vae.device

        self.eta_theta = None

    def save_model(self):
        """save model to disk"""
        path = pathlib.Path().resolve()

        # network
        self.vae.save_model()

        # global parameters
        torch.save(self.eta_theta, os.path.join(path, f"eta_theta.pt"))

    def load_model(self):
        """load model from disk"""
        path = pathlib.Path().resolve()

        # network
        self.vae.load_model()

        # global parameters
        self.eta_theta = torch.load(os.path.join(path, f"eta_theta.pt"))

    def encode(self, y):
        mu, log_var = self.vae.encode(y)
        # scale should be positive, and thus it's negative inverse should be negative
        scale = -torch.exp(0.5 * log_var)
        potentials = pack_dense(scale, mu)
        return potentials

    def decode(self, x):
        return self.vae.decode(x)

    def forward(self, y):
        potentials = self.encode(y)
        eta_x, label_stats, _, _ = local_optimization(potentials, self.eta_theta)
        classes = torch.argmax(label_stats, dim=-1)
        x = Gaussian(eta_x).rsample()
        mu_y, log_var_y = self.decode(x)
        return mu_y, log_var_y, x, classes

    def save_and_log(self, obs, epoch, save_path, eta_theta):
        with torch.no_grad():
            data = torch.tensor(obs).to(self.vae.device).float()
            potentials = self.encode(data)

            eta_x, label_stats, _, _ = local_optimization(potentials, eta_theta)

            # get encoded means
            gaussian_stats = Gaussian(eta_x).expected_stats()
            _, Ex, _, _ = unpack_dense(gaussian_stats)

            # get reconstructions
            x = Gaussian(eta_x).rsample()
            mu_y, log_var_y = self.decode(x)

            plot_reconstruction(
                obs=obs,
                mu=mu_y.cpu().detach().numpy(),
                log_var=log_var_y.cpu().detach().numpy(),
                latent=Ex.cpu().detach().numpy(),
                eta_theta=eta_theta,
                classes=torch.argmax(label_stats, dim=-1).cpu().detach().numpy(),
                title=f"epoch:{epoch}_svae",
                save_path=save_path,
            )

    def fit(self, obs, epochs, batch_size, K, kld_weight, save_path=None):
        """
        Find the optimum for global variational parameter eta_theta, and encoder/decoder parameters.

        Parameters
        ----------
        obs:
            Observations
        epochs:
            Number of epochs to train.
        batch_size:
            Size of each batch.
        K:
            Number of clusters in latent space.
        kld_weight:
            Weight for the KLD in the loss.
        save_path:
            Where to save plots etc.
        """
        print("Training the SVAE ...")

        if save_path is not None:
            os.mkdir(save_path)

        # Make data object
        data = torch.tensor(obs).to(self.vae.device)
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True
        )
        num_batches = len(dataloader)

        _, D = data.shape
        eta_theta_prior = initialize_global_parameters(
            K, D, alpha=0.05 / K, niw_conc=1.0, random_scale=0.0
        )
        eta_theta = initialize_global_parameters(
            K, D, alpha=1.0, niw_conc=1.0, random_scale=3.0
        )

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3, weight_decay=0.001)

        train_loss = []
        # self.save_and_log(obs, "pre", save_path, eta_theta)
        for epoch in tqdm(range(epochs + 1)):

            total_loss = []
            for i, y in enumerate(dataloader):
                y = y.float()
                potentials = self.encode(y)

                # remove dependency on previous iterations
                eta_theta = (eta_theta[0].detach(), eta_theta[1].detach())

                """
                Find local optimum for local variational parameters eta_x, eta_z
                """
                eta_x, _, prior_stats, local_kld = local_optimization(
                    potentials, eta_theta
                )

                # get the latents using the eta_x parameters we just optimized
                x = Gaussian(eta_x).rsample()

                """
                Update global variational parameter eta_theta using natural gradient
                """
                nat_grad = natural_gradient(
                    prior_stats, eta_theta, eta_theta_prior, len(obs), num_batches
                )

                # do SGD on the natural gradient
                step_size = 10
                eta_theta = tuple(
                    [
                        eta_theta[i] - step_size * nat_grad[i]
                        for i in range(len(eta_theta))
                    ]
                )

                """
                Update encoder/decoder parameters using automatic differentiation
                """
                # reconstruction loss
                mu_y, log_var_y = self.decode(x)
                recon_loss = num_batches * self.vae.loss_function(y, mu_y, log_var_y)

                # regularization
                global_kld = prior_kld(eta_theta, eta_theta_prior)
                kld_loss = (global_kld + num_batches * local_kld) / len(y)

                # loss is a combination of above two
                loss = recon_loss + kld_weight * kld_loss

                optimizer.zero_grad()
                # compute gradients
                loss.backward()
                # update parameters
                optimizer.step()

                total_loss.append((recon_loss.item(), kld_weight * kld_loss))
            train_loss.append(np.mean(total_loss, axis=0))

            if epoch % max((epochs // 10), 1) == 0:
                self.save_and_log(obs, epoch, save_path, eta_theta)

        self.save_and_log(obs, "end", save_path, eta_theta)

        print("Finished training of the SVAE")
        self.eta_theta = eta_theta
        self.save_model()
        return train_loss
