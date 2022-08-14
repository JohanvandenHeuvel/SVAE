import os
import pathlib

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from distributions.dense import pack_dense, unpack_dense
from svae.gradient import natural_gradient, gradient_descent
from svae.lds.global_optimization import initialize_global_lds_parameters, prior_kld_lds
from svae.lds.local_optimization import local_optimization
from vae import VAE

plt.ion()
fig, axs = plt.subplots(3, 5, figsize=(5 * 10, 10))
fig.tight_layout()
plt.show()


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
        torch.save(self.eta_theta, os.path.join(path, f"../eta_theta.pt"))

    def load_model(self):
        """load model from disk"""
        path = pathlib.Path().resolve()

        # network
        self.vae.load_model()

        # global parameters
        self.eta_theta = torch.load(os.path.join(path, f"../eta_theta.pt"))

    def encode(self, y):
        mu, log_var = self.vae.encode(y)
        # scale should be positive, and thus it's negative inverse should be negative
        # scale = -torch.exp(0.5 * log_var)
        scale = -0.5 * torch.log1p(torch.exp(log_var))
        potentials = pack_dense(scale, mu)
        return potentials

    def decode(self, x):
        mu_y, log_var_y = self.vae.decode(x)
        return torch.sigmoid(mu_y), torch.log1p(log_var_y.exp())
        # return torch.sigmoid(mu_y), log_var_y

    def forward(self, y):
        potentials = self.encode(y)
        x, eta_x, _, _ = local_optimization(potentials, self.eta_theta)
        mu_y, log_var_y = self.decode(x)
        return mu_y, log_var_y, x

    # def save_and_log(self, obs, epoch, save_path, eta_theta):
    #
    #     def zero_out(prefix, potentials):
    #         scale, loc, _, _ = unpack_dense(potentials)
    #         loc[prefix:] = 0.0
    #         scale[prefix:] = 0.0
    #         potentials = pack_dense(scale, loc)
    #         return potentials
    #
    #     def get_big_image(n_samples=5):
    #         samples = []
    #         for i in range(5):
    #             # samples
    #             sample, _, _, _ = local_optimization(potentials, eta_theta)
    #             # reconstruction
    #             y, _ = self.decode(sample.squeeze())
    #             # save
    #             samples.append(y)
    #         mean_image = torch.stack(samples).mean(0)
    #         samples = torch.hstack(samples[:n_samples])
    #
    #         big_image = torch.hstack((data, mean_image, samples))
    #         big_image = big_image.cpu().detach().numpy()
    #         return big_image
    #
    #     with torch.no_grad():
    #         # only use a subset of the data for plotting
    #         data = torch.tensor(obs).to(self.vae.device).float()
    #         data = data[:100]
    #
    #         # set the observations to zero after prefix
    #         prefix = 25
    #         potentials = self.encode(data)
    #         potentials = zero_out(prefix, potentials)
    #
    #         big_image = get_big_image()
    #
    #         fig, ax = plt.subplots(figsize=(10, 10))
    #         ax.matshow(big_image, cmap="gray")
    #         ax.plot([-0.5, big_image.shape[1]], [prefix-0.5, prefix-0.5], 'r', linewidth=2)
    #         ax.autoscale(False)
    #         ax.axis("off")
    #
    #         fig.tight_layout()
    #         plt.show()

    def save_and_log(self, obs, epoch, save_path, eta_theta):
        def zero_out(prefix, potentials):
            scale, loc, _, _ = unpack_dense(potentials)
            loc[prefix:] = 0.0
            scale[prefix:] = 0.0
            potentials = pack_dense(scale, loc)
            return potentials

        def get_samples(n_samples=5):
            samples = []
            latent_samples = []
            for i in range(n_samples):
                # samples
                sample, _, _, _ = local_optimization(potentials, eta_theta)
                sample = sample.squeeze()
                # reconstruction
                y, _ = self.decode(sample)
                # save
                samples.append(y)
                latent_samples.append(sample)

            return samples, latent_samples

        with torch.no_grad():
            # only use a subset of the data for plotting
            data = torch.tensor(obs).to(self.vae.device).float()
            data = data[:100]

            # set the observations to zero after prefix
            prefix = 25
            potentials = self.encode(data)
            potentials = zero_out(prefix, potentials)

            # get samples
            n_samples = 5
            samples, latent_samples = get_samples(n_samples)

            for i in range(n_samples):

                ax = axs[:, i]

                ax[0].clear()
                ax[0].matshow(data.T.cpu().detach().numpy(), cmap="gray")
                ax[0].plot(
                    [prefix - 0.5, prefix - 0.5],
                    [-0.5, data.shape[1]],
                    "r",
                    linewidth=2,
                )
                ax[0].axis("off")

                ax[1].clear()
                ax[1].matshow(samples[i].T.cpu().detach().numpy(), cmap="gray")
                ax[1].plot(
                    [prefix - 0.5, prefix - 0.5],
                    [-0.5, data.shape[1]],
                    "r",
                    linewidth=2,
                )
                ax[1].axis("off")

                ax[2].clear()
                for latent_state in latent_samples[i].T:
                    ax[2].plot(latent_state.cpu().detach().numpy())
                # ax[2].plot([prefix-0.5, prefix-0.5], [-0.5, data.shape[1]], 'r', linewidth=2)
                # ax[2].axis("off")
                # ax[2].autoscale(False)

    def fit(self, obs, epochs, batch_size, latent_dim, kld_weight, save_path=None):
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
        save_path:
            Where to save plots etc.
        """
        print("Training the SVAE ...")

        if save_path is not None:
            os.mkdir(save_path)

        # Make data object
        data = torch.tensor(obs).to(self.vae.device)
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False
        )
        num_batches = len(dataloader)

        niw_prior, mniw_prior = initialize_global_lds_parameters(latent_dim)
        niw_param, mniw_param = initialize_global_lds_parameters(latent_dim)

        mniw_prior, mniw_param = list(mniw_prior), list(mniw_param)

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-4)

        train_loss = []
        self.save_and_log(obs, "pre", save_path, (niw_param, mniw_param))
        fig.canvas.draw()
        plt.pause(0.1)
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
                x, _, (E_init_stats, E_pair_stats), local_kld = local_optimization(
                    potentials, (niw_param, mniw_param)
                )

                """
                Update global variational parameter eta_theta using natural gradient
                """
                # update global param
                # nat_grad_init = natural_gradient(
                #     pack_dense(*E_init_stats), niw_param, niw_prior, len(data), num_batches
                # )
                # niw_param = gradient_descent(
                #     niw_param, torch.stack(nat_grad_init), step_size=1e-1
                # )

                # nat_grad_pair = natural_gradient(
                #     E_pair_stats, mniw_param, mniw_prior, len(data), num_batches
                # )
                # mniw_param = gradient_descent(mniw_param, nat_grad_pair, step_size=1e-5)

                """
                Update encoder/decoder parameters using automatic differentiation
                """
                # reconstruction loss
                mu_y, log_var_y = self.decode(x)
                recon_loss = num_batches * self.vae.loss_function(y, mu_y, log_var_y)

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

                total_loss.append((recon_loss.item(), kld_weight * kld_loss))
                # print(f"{i}: {total_loss[-1]}")

                # if epoch % max((epochs // 10), 1) == 0 or epoch == 0:
                #     print(total_loss[-1])
            train_loss.append(np.mean(total_loss, axis=0))
            print(f"{epoch}: {train_loss[-1]}")

            # if epoch % max((epochs // 10), 1) == 0:
            #     self.save_and_log(obs, epoch, save_path, (niw_param, mniw_param))
            self.save_and_log(obs, epoch, save_path, (niw_param, mniw_param))
            fig.canvas.draw()
            plt.pause(0.1)

        # self.save_and_log(obs, "end", save_path, eta_theta)

        print("Finished training of the SVAE")
        # self.eta_theta = eta_theta
        self.save_model()
        return train_loss
