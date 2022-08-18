import os
import pathlib
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from distributions.dense import pack_dense, unpack_dense
from svae.gradient import natural_gradient, gradient_descent
from svae.lds.global_optimization import initialize_global_lds_parameters, prior_kld_lds
from svae.lds.local_optimization import local_optimization
from vae import VAE
from plot.lds_plot import plot


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
        # scale = -0.5 * torch.log1p(torch.exp(log_var))
        potentials = pack_dense(scale, mu)
        return potentials

    def decode(self, x):
        mu_y, log_var_y = self.vae.decode(x)
        # return torch.sigmoid(mu_y), torch.log1p(log_var_y.exp())
        return torch.sigmoid(mu_y), log_var_y
        # return mu_y, log_var_y

    def forward(self, y):
        potentials = self.encode(y)
        x, eta_x, _, _ = local_optimization(potentials, self.eta_theta)
        mu_y, log_var_y = self.decode(x)
        return mu_y, log_var_y, x

    def save_and_log(self, obs, epoch, eta_theta):
        def zero_out(prefix, potentials):
            """Zero out a part of the data.

            Parameters
            ----------
            prefix: int
                After which time step zero-out the data.
            potentials: tensor
                Output from the encoder network.
            """
            scale, loc, _, _ = unpack_dense(potentials)
            loc[prefix:] = 0.0
            scale[prefix:] = 0.0
            potentials = pack_dense(scale, loc)
            return potentials

        def get_samples(n_samples=5):
            """Get decoded samples.

            Parameters
            ----------
            n_samples: int
                Number of samples.
            """
            decoded_samples = []
            latent_samples = []
            latent_means = []
            latent_vars = []
            for i in range(n_samples):
                # samples
                sample, _, _, _, (mean, variance) = local_optimization(
                    potentials, eta_theta
                )
                sample = sample.squeeze()
                # reconstruction
                y, _ = self.decode(sample)
                # save
                decoded_samples.append(y)
                latent_samples.append(sample)
                latent_means.append(mean)
                latent_vars.append(variance)

            decoded_samples = torch.stack(decoded_samples)
            latent_samples = torch.stack(latent_samples)
            latent_means = torch.stack(latent_means)
            latent_vars = torch.stack(latent_vars)

            return decoded_samples, latent_samples, latent_means, latent_vars

        with torch.no_grad():

            self.eta_theta = eta_theta
            self.save_model(epoch)

            # only use a subset of the data for plotting
            data = torch.tensor(obs).to(self.vae.device).float()
            data = data[:100]

            # set the observations to zero after prefix
            prefix = 25
            potentials = self.encode(data)
            potentials = zero_out(prefix, potentials)

            # get samples
            n_samples = 5
            samples, latent_samples, latent_means, latent_vars = get_samples(n_samples)
            plot(
                obs=data.cpu().detach().numpy(),
                samples=samples.cpu().detach().numpy(),
                latent_samples=latent_samples.cpu().detach().numpy(),
                latent_means=latent_means.cpu().detach().numpy(),
                latent_vars=latent_vars.cpu().detach().numpy(),
                title=f"epoch:{epoch}",
                save_path=self.save_path,
            )

    def fit(self, obs, epochs, batch_size, latent_dim, kld_weight):
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
        kld_weight:
            Weight for the KLD in the loss.
        """
        print("Training the SVAE ...")

        # Make data object
        data = torch.tensor(obs).to(self.vae.device)
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False
        )
        num_batches = len(dataloader)

        niw_prior, mniw_prior = initialize_global_lds_parameters(latent_dim)
        niw_param, mniw_param = initialize_global_lds_parameters(latent_dim)

        mniw_prior, mniw_param = list(mniw_prior), list(mniw_param)

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3, weight_decay=1e-2)

        train_loss = []
        self.save_and_log(obs, "pre", (niw_param, mniw_param))
        try:
            for epoch in tqdm(range(epochs + 1)):

                total_loss = []
                for i, y in enumerate(dataloader):
                    y = y.float()
                    potentials = self.encode(y)

                    # remove dependency on previous iterations
                    niw_param = niw_param.detach()
                    mniw_param = tuple([p.detach() for p in mniw_param])

                    """
                    Find local optimum for local variational parameters eta_x, eta_z
                    """
                    (
                        x,
                        _,
                        (E_init_stats, E_pair_stats),
                        local_kld,
                        _,
                    ) = local_optimization(potentials, (niw_param, mniw_param))

                    """
                    Update global variational parameter eta_theta using natural gradient
                    """
                    # update global param
                    nat_grad_init = natural_gradient(
                        pack_dense(*E_init_stats),
                        niw_param,
                        niw_prior,
                        len(data),
                        num_batches,
                    )
                    niw_param = gradient_descent(
                        niw_param, torch.stack(nat_grad_init), step_size=1e-1
                    )

                    nat_grad_pair = natural_gradient(
                        E_pair_stats, mniw_param, mniw_prior, len(data), num_batches
                    )
                    mniw_param = gradient_descent(
                        mniw_param, nat_grad_pair, step_size=1e-5
                    )

                    """
                    Update encoder/decoder parameters using automatic differentiation
                    """
                    # reconstruction loss
                    mu_y, log_var_y = self.decode(x)
                    recon_loss = num_batches * self.vae.loss_function(
                        y, mu_y.squeeze(), log_var_y.squeeze()
                    )

                    # regularization
                    # global_kld = prior_kld_lds((niw_param, mniw_param), (niw_prior, mniw_prior))
                    global_kld = 0.0
                    kld_loss = (global_kld + local_kld) / len(y)

                    loss = recon_loss + kld_weight * kld_loss

                    optimizer.zero_grad()
                    # compute gradients
                    loss.backward()
                    # update parameters
                    optimizer.step()

                    total_loss.append((recon_loss.item(), kld_weight * kld_loss.item()))

                train_loss.append(np.mean(total_loss, axis=0))
                print(f"{epoch}: {train_loss[-1]}")

                if epoch % max((epochs // 20), 1) == 0:
                    self.save_and_log(obs, epoch, (niw_param, mniw_param))

        finally:
            print("Finished training of the SVAE")
            self.save_and_log(obs, "end", (niw_param, mniw_param))
            return train_loss
