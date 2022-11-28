import os

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from distributions import Gaussian
from matrix_ops import pack_dense, unpack_dense
from plot.gmm_plot import plot_reconstruction
from svae.gradient import natural_gradient, SGDOptim
from vae import VAE
from .global_optimization import (
    initialize_global_gmm_parameters,
    prior_kld_gmm,
)
from .local_optimization import local_optimization


class SVAE:
    def __init__(self, vae: VAE, save_path=None):
        self.vae = vae
        self.device = vae.device

        self.eta_theta = None
        self.save_path = save_path

        if save_path is not None:
            os.mkdir(save_path)

    def save_model(self, epoch):
        """save model to disk"""
        path = self.save_path

        # network
        self.vae.save_model(path, epoch)

        # global parameters
        torch.save(self.eta_theta, os.path.join(path, f"eta_theta_{epoch}.pt"))

    def load_model(self, path, epoch):
        """load model from disk"""

        # network
        self.vae.load_model(path, epoch)

        # global parameters
        self.eta_theta = torch.load(os.path.join(path, f"eta_theta_{epoch}.pt"))

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
        x, eta_x, label_stats, _, _ = local_optimization(potentials, self.eta_theta)
        classes = torch.argmax(label_stats, dim=-1)
        mu_y, log_var_y = self.decode(x)
        return mu_y, log_var_y, x, classes

    def save_and_log(self, obs, epoch, eta_theta):
        with torch.no_grad():
            self.eta_theta = eta_theta

            if self.save_path is not None:
                self.save_model(epoch)

            data = torch.tensor(obs).to(self.vae.device).double()
            potentials = self.encode(data)

            x, eta_x, label_stats, _, _ = local_optimization(potentials, eta_theta)

            # get encoded means
            gaussian_stats = Gaussian(eta_x).expected_stats()
            _, Ex, _, _ = unpack_dense(gaussian_stats)

            # get reconstructions
            mu_y, log_var_y = self.decode(x)

            fig = plot_reconstruction(
                obs=obs,
                mu=mu_y.squeeze().cpu().detach().numpy(),
                latent=Ex.cpu().detach().numpy(),
                eta_theta=eta_theta,
                classes=torch.argmax(label_stats, dim=-1).cpu().detach().numpy(),
            )

            wandb.log({"fig": fig})

    def fit(self, obs, epochs, batch_size, K, kld_weight, train=True, verbose=True):
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
        """
        print("Training the SVAE...")

        # Make data object
        data = torch.tensor(obs).to(self.vae.device)
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True
        )
        num_batches = len(dataloader)

        _, D = data.shape
        eta_theta_prior = initialize_global_gmm_parameters(
            K, D, alpha=0.05 / K, niw_conc=1.0, random_scale=0.0
        )
        eta_theta = initialize_global_gmm_parameters(
            K, D, alpha=1.0, niw_conc=1.0, random_scale=3.0
        )

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3, weight_decay=1e-3)
        global_optimizer = SGDOptim(step_size=10)

        train_loss = []
        if verbose:
            self.save_and_log(obs, "pre", eta_theta)

        for epoch in range(epochs + 1):

            total_loss = []
            for i, y in enumerate(dataloader):
                y = y.double()
                potentials = self.encode(y)

                # remove dependency on previous iterations
                eta_theta = (eta_theta[0].detach(), eta_theta[1].detach())

                """
                Find local optimum for local variational parameters eta_x, eta_z
                """
                x, _, _, prior_stats, local_kld = local_optimization(
                    potentials, eta_theta
                )

                """
                Update global variational parameter eta_theta using natural gradient
                """
                nat_grad = natural_gradient(
                    prior_stats, eta_theta, eta_theta_prior, len(obs), num_batches
                )

                # do SGD on the natural gradient
                eta_theta = tuple(
                    [
                        global_optimizer.update(eta_theta[i], nat_grad[i])
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
                global_kld = prior_kld_gmm(eta_theta, eta_theta_prior)
                kld_loss = (global_kld + num_batches * local_kld) / len(y)

                # loss is a combination of above two
                loss = recon_loss + kld_weight * kld_loss

                optimizer.zero_grad()
                # compute gradients
                loss.backward()
                # update parameters
                optimizer.step()

                wandb.log({"recon_loss": recon_loss, "kld": kld_weight * kld_loss})

                total_loss.append((recon_loss.item(), kld_weight * kld_loss.item()))
            train_loss.append(np.mean(total_loss, axis=0))

            if verbose:
                if epoch % max((epochs // 10), 1) == 0:
                    print(f"[{epoch}/{epochs + 1}] {train_loss[-1].sum()}")
                    self.save_and_log(obs, epoch, eta_theta)

        print("Finished training of the SVAE")
        # TODO otherwise if verbose == False this is not set correctly
        self.eta_theta = eta_theta
        if verbose:
            self.save_and_log(obs, "end", eta_theta)
