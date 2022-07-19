import os
import pathlib
from typing import Tuple

import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dense import pack_dense, unpack_dense
from plot import lds_plot

from svae.local_optimization.lds import local_optimization
from svae.global_optimization import (
    natural_gradient,
    initialize_global_lds_parameters,
    prior_kld_lds,
)
from vae import VAE


def gradient_descent(w, grad_w, step_size):
    if isinstance(w, Tuple):
        return [w[i] - step_size * grad_w[i] for i in range(len(w))]
    return w - step_size * grad_w


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
        x, eta_x, _, _ = local_optimization(potentials, self.eta_theta)
        mu_y, log_var_y = self.decode(x)
        return mu_y, log_var_y, x

    def save_and_log(self, obs, epoch, save_path, eta_theta):
        with torch.no_grad():
            # only use a subset of the data for plotting
            data = torch.tensor(obs).to(self.vae.device).float()
            data = data[:100]

            # set the observations to zero after prefix
            prefix = 25
            potentials = self.encode(data)
            scale, loc, _, _ = unpack_dense(potentials)
            loc[prefix:] = 0.0
            scale[prefix:] = 0.0
            potentials = pack_dense(scale, loc)

            # get samples
            samples = []
            for i in range(5):
                # samples
                sample, _, _, _ = local_optimization(potentials, eta_theta, num_samples=1)
                # reconstruction
                y, _ = self.decode(sample.squeeze())
                # save
                samples.append(y)
            mean_image = torch.stack(samples).mean(0)
            samples = torch.hstack(samples)

            big_image = torch.hstack((data, mean_image, samples))
            big_image = big_image.cpu().detach().numpy()

            fig, ax = plt.subplots()
            ax.matshow(big_image, cmap="gray")
            ax.plot([-0.5, big_image.shape[1]], [prefix-0.5, prefix-0.5], 'r', linewidth=2)
            ax.autoscale(False)
            ax.axis("off")

            fig.tight_layout()
            plt.show()

            print("Fuck yeah!")

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
        eta_theta_prior = initialize_global_lds_parameters(10)
        eta_theta = initialize_global_lds_parameters(10)

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3, weight_decay=0.001)

        train_loss = []
        self.save_and_log(obs, "pre", save_path, eta_theta)
        for epoch in tqdm(range(epochs + 1)):

            total_loss = []
            for i, y in enumerate(dataloader):
                y = y.float()
                potentials = self.encode(y)

                # remove dependency on previous iterations
                eta_theta = (eta_theta[0].detach(), tuple([e.detach() for e in eta_theta[1]]))

                """
                Find local optimum for local variational parameters eta_x, eta_z
                """
                x, eta_x, prior_stats, local_kld = local_optimization(
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
                        gradient_descent(eta_theta[i], nat_grad[i], step_size=1)
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
                # global_kld = prior_kld_gmm(eta_theta, eta_theta_prior)
                # global_kld = prior_kld_lds(eta_theta, eta_theta_prior, eta_x)
                # kld_loss = (global_kld + num_batches * local_kld) / len(y)

                # loss is a combination of above two
                # loss = recon_loss + kld_weight * kld_loss
                loss = recon_loss

                optimizer.zero_grad()
                # compute gradients
                loss.backward()
                # update parameters
                optimizer.step()

                # total_loss.append((recon_loss.item(), kld_weight * kld_loss))
                total_loss.append((recon_loss.item()))
            train_loss.append(np.mean(total_loss, axis=0))

            # if epoch % max((epochs // 10), 1) == 0:
            #     self.save_and_log(obs, epoch, save_path, eta_theta)

        # self.save_and_log(obs, "end", save_path, eta_theta)

        print("Finished training of the SVAE")
        self.eta_theta = eta_theta
        self.save_model()
        return train_loss
