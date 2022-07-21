import os
import pathlib

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dense import pack_dense
from svae.global_optimization import (
    natural_gradient,
    initialize_global_lds_parameters,
    gradient_descent,
)
from svae.local_optimization.lds import local_optimization
from vae import VAE


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
        # scale = -0.5 * torch.log1p(torch.exp(log_var))
        potentials = pack_dense(scale, mu)
        return potentials

    def decode(self, x):
        return self.vae.decode(x)

    def forward(self, y):
        potentials = self.encode(y)
        x, eta_x, _, _ = local_optimization(potentials, self.eta_theta)
        mu_y, log_var_y = self.decode(x)
        return mu_y, log_var_y, x

    # def save_and_log(self, obs, epoch, save_path, eta_theta):
    #     with torch.no_grad():
    #         # only use a subset of the data for plotting
    #         data = torch.tensor(obs).to(self.vae.device).float()
    #         data = data[:100]
    #
    #         # # set the observations to zero after prefix
    #         prefix = 25
    #         potentials = self.encode(data)
    #         # scale, loc, _, _ = unpack_dense(potentials)
    #         # loc[prefix:] = 0.0
    #         # scale[prefix:] = 0.0
    #         # potentials = pack_dense(scale, loc)
    #
    #         # get samples
    #         samples = []
    #         for i in range(5):
    #             # samples
    #             sample = lds_sample(eta_theta, potentials, num_samples=1)
    #             # reconstruction
    #             y, _ = self.decode(sample.squeeze())
    #             # save
    #             samples.append(y)
    #         mean_image = torch.stack(samples).mean(0)
    #         samples = torch.hstack(samples)
    #
    #         big_image = torch.hstack((data, mean_image, samples))
    #         big_image = big_image.cpu().detach().numpy()
    #
    #         fig, ax = plt.subplots()
    #         ax.matshow(big_image, cmap="gray")
    #         ax.plot([-0.5, big_image.shape[1]], [prefix-0.5, prefix-0.5], 'r', linewidth=2)
    #         ax.autoscale(False)
    #         ax.axis("off")
    #
    #         fig.tight_layout()
    #         plt.show()
    #
    #         print("Fuck yeah!")

    def fit(self, obs, epochs, batch_size, save_path=None):
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

        latents, data = obs

        # Make data object
        data = data.unsqueeze(1).to(self.vae.device)
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False
        )

        niw_prior, mniw_prior = initialize_global_lds_parameters(1)
        niw_param, mniw_param = initialize_global_lds_parameters(1)

        mniw_prior, mniw_param = list(mniw_prior), list(mniw_param)

        # optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3, weight_decay=0.001)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)

        train_loss = []
        # self.save_and_log(obs, "pre", save_path, eta_theta)
        for epoch in tqdm(range(epochs + 1)):

            total_loss = []
            for _, y in enumerate(dataloader):
                # y = y.float().
                potentials = self.encode(y)

                # remove dependency on previous iterations
                niw_param = niw_param.detach()
                mniw_param = tuple([p.detach() for p in mniw_param])

                """
                Find local optimum for local variational parameters eta_x, eta_z
                """
                if epoch % max((epochs // 10), 1) == 0 or epoch == 0:
                    plot = True
                else:
                    plot = False
                x, _, (E_init_stats, E_pair_stats), _ = local_optimization(
                    potentials, (niw_param, mniw_param), latents, y, plot
                )

                """
                Update global variational parameter eta_theta using natural gradient
                """
                # update global param
                nat_grad_init = natural_gradient(
                    E_init_stats, niw_param, niw_prior, len(data), 1
                )
                niw_param = gradient_descent(
                    niw_param, torch.stack(nat_grad_init), step_size=1e-1
                )

                nat_grad_pair = natural_gradient(
                    E_pair_stats, mniw_param, mniw_prior, len(data), 1
                )
                mniw_param = gradient_descent(mniw_param, nat_grad_pair, step_size=1e-2)

                """
                Update encoder/decoder parameters using automatic differentiation
                """
                # reconstruction loss
                mu_y, log_var_y = self.decode(x)
                recon_loss = self.vae.loss_function(y, mu_y, log_var_y)

                if plot:
                    plt.plot(y.cpu().detach().numpy())
                    plt.plot(
                        self.vae.reparameterize(mu_y, log_var_y)
                        .squeeze()
                        .cpu()
                        .detach()
                        .numpy()
                    )
                    plt.show()

                loss = recon_loss

                optimizer.zero_grad()
                # compute gradients
                loss.backward()
                # update parameters
                optimizer.step()

                # total_loss.append((recon_loss.item()))
            # train_loss.append(np.mean(total_loss, axis=0))

            # if epoch % max((epochs // 10), 1) == 0:
            #     self.save_and_log(obs, epoch, save_path, eta_theta)
            # self.save_and_log(obs, epoch, save_path, eta_theta)

        # self.save_and_log(obs, "end", save_path, eta_theta)

        print("Finished training of the SVAE")
        # self.eta_theta = eta_theta
        self.save_model()
        return train_loss
