import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from distributions.gaussian import natural_to_info
from matrix_ops import pack_dense, unpack_dense
from plot.lds_plot import (
    plot_observations,
    plot,
    plot_latents,
    plot_parameters,
    plot_info_parameters,
    plot_potentials,
)
from svae.gradient import natural_gradient, SGDOptim
from svae.lds.global_optimization import initialize_global_lds_parameters, prior_kld_lds
from svae.lds.local_optimization import local_optimization, standard_pair_params
from vae import VAE

from distributions import MatrixNormalInverseWishart, NormalInverseWishart, Gaussian

np.set_printoptions(
    edgeitems=30,
    linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x),
    precision=2,
)


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

    def encode(self, y, tanh=True):
        mu, log_var = self.vae.encode(y)

        if tanh:
            scale = -0.5 * torch.exp(torch.tanh(log_var / 10) * 10)
        else:
            # scale should be positive, and thus it's negative inverse should be negative
            scale = -torch.exp(0.5 * log_var)
        # scale = -0.5 * torch.log1p(torch.exp(log_var))
        potentials = pack_dense(scale, mu)
        return potentials

    def decode(self, x, sigmoid=True, tanh=True):
        mu_y, log_var_y = self.vae.decode(x)
        # return torch.sigmoid(mu_y), torch.log1p(log_var_y.exp())
        if sigmoid:
            mu_y = torch.sigmoid(mu_y)
        if tanh:
            log_var_y = torch.tanh(log_var_y / 10) * 10
        return mu_y, log_var_y

    def forward(self, y):
        potentials = self.encode(y)
        x, _, _ = local_optimization(potentials, self.eta_theta)
        mu_y, log_var_y = self.decode(x)
        return mu_y, log_var_y, x

    def save_and_log(self, data, epoch, eta_theta):
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
            x, _, _ = local_optimization(potentials, eta_theta, n_samples)
            mu_y, log_var_y = self.decode(x.reshape(-1, x.shape[-1]))

            mu_y = mu_y.reshape(*x.shape[:-1], -1)
            mu_y = torch.swapaxes(mu_y, axis0=0, axis1=1)

            log_var_y = log_var_y.reshape(*x.shape[:-1], -1)
            log_var_y = torch.swapaxes(log_var_y, axis0=0, axis1=1)

            x = torch.swapaxes(x, axis0=0, axis1=1)
            return mu_y, log_var_y, x

        with torch.no_grad():
            self.eta_theta = eta_theta
            niw_param, mniw_param = eta_theta
            self.save_model(epoch)

            J11, J12, J22, _ = MatrixNormalInverseWishart(mniw_param).expected_stats()
            A, Q = standard_pair_params(-2 * J11, -1 * J12, -2 * J22)
            Sigma, mu, _, _ = unpack_dense(
                NormalInverseWishart(niw_param).expected_stats()
            )
            # print(sorted(torch.abs(torch.linalg.eigvalsh(A)).cpu().detach().numpy(), reverse=True))

            # only use a subset of the data for plotting
            data = data[:200]

            # set the observations to zero after prefix
            prefix = 100
            potentials = self.encode(data)
            potentials = zero_out(prefix, potentials)

            # get samples
            n_samples = 50
            decoded_means, decoded_vars, latent_samples = get_samples(n_samples)

            plot_potentials(
                potentials,
                prefix=prefix,
                title=f"{epoch}_potentials",
                save_path=self.save_path,
            )

            plot_parameters(
                A, Q, Sigma, mu, title=f"{epoch}_params", save_path=self.save_path
            )

            plot_info_parameters(
                -1 * J11,
                -2 * J12,
                -2 * J22,
                -2 * J12.T,
                title=f"{epoch}_info_params",
                save_path=self.save_path,
            )

            plot(
                obs=data.cpu().detach().numpy(),
                samples=decoded_means.cpu().detach().numpy(),
                prefix=prefix,
                title=f"{epoch}_obs",
                save_path=self.save_path,
            )

            plot_latents(
                latents=latent_samples.cpu().detach().numpy(),
                prefix=prefix,
                title=f"{epoch}_latents",
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
        latent_dim:
            Size of the latent dimension
        kld_weight:
            Weight for the KLD in the loss.
        """
        print("Training the SVAE ...")

        """
        Data setup 
        """
        if not isinstance(obs, torch.Tensor):
            data = torch.tensor(obs)
        else:
            data = obs.clone().detach()
        data = data.to(self.vae.device).double()
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False
        )
        num_batches = len(dataloader)

        """
        Initialize priors 
        """
        niw_prior, mniw_prior = initialize_global_lds_parameters(latent_dim)
        niw_param, mniw_param = initialize_global_lds_parameters(latent_dim)
        mniw_prior, mniw_param = list(mniw_prior), list(mniw_param)

        """
        Optimizer setup 
        """
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        niw_optimizer = SGDOptim(step_size=1)
        mniw_optimizer = [
            SGDOptim(step_size=1),
            SGDOptim(step_size=1),
            SGDOptim(step_size=1),
            SGDOptim(step_size=1),
        ]

        """
        Optimization loop 
        """
        self.save_and_log(data, "pre", (niw_param, mniw_param))
        train_loss = []
        for epoch in range(epochs + 1):

            # print(f"> EPOCH {epoch} =====")

            total_loss = []
            for i, y in enumerate(dataloader):

                if i >= len(dataloader) - 1:
                    continue

                # print(f">> ITER {i} =====")
                potentials = self.encode(y)

                # remove dependency on previous iterations
                niw_param = niw_param.detach()
                niw_param.requires_grad = True
                mniw_param = list([p.detach() for p in mniw_param])
                for p in mniw_param:
                    p.requires_grad = True

                """
                Find local optimum for local variational parameters eta_x, eta_z
                """
                x, (E_init_stats, E_pair_stats), local_kld = local_optimization(
                    potentials, (niw_param, mniw_param), n_samples=10
                )

                # regularization
                global_kld = prior_kld_lds(
                    (niw_param, mniw_param), (niw_prior, mniw_prior)
                )

                """
                Update global variational parameter eta_theta using natural gradient
                """
                # update global param
                # nat_grad_init = natural_gradient(
                #     pack_dense(*E_init_stats)[None, ...],
                #     niw_param,
                #     niw_prior,
                #     len(data),
                #     num_batches,
                # )
                # niw_param = niw_optimizer.update(niw_param, torch.stack(nat_grad_init))

                nat_grad_pair = natural_gradient(
                    E_pair_stats, mniw_param, mniw_prior, len(data), num_batches
                )
                mniw_param = [
                    mniw_optimizer[i].update(mniw_param[i], nat_grad_pair[i])
                    for i in range(len(nat_grad_pair))
                ]

                # print(nat_grad_pair[0].cpu().detach().numpy())
                # print()
                # print(nat_grad_pair[1].cpu().detach().numpy())
                # print()
                # print(nat_grad_pair[2].cpu().detach().numpy())
                # print()
                # print(nat_grad_pair[3])

                """
                Update encoder/decoder parameters using automatic differentiation
                """
                # reconstruction loss
                mu_y, log_var_y = self.decode(x.reshape(-1, x.shape[-1]))
                recon_loss = (
                    self.vae.loss_function(
                        y[:, None, :],
                        mu_y.reshape(*x.shape[:-1], -1),
                        log_var_y.reshape(*x.shape[:-1], -1),
                        full=True,
                        reduction="sum",
                    )
                    / x.shape[1]
                )

                recon_loss = (num_batches * recon_loss) / len(data)
                local_kld = (num_batches * local_kld) / len(data)
                global_kld = global_kld / len(data)

                kld_loss = global_kld + local_kld

                loss = recon_loss + kld_weight * kld_loss
                # print(f"{global_kld:.3f} \t {local_kld.item():.3f} \t {recon_loss:.3f}")

                optimizer.zero_grad()
                # compute gradients
                loss.backward()
                # update parameters
                optimizer.step()

                total_loss.append(
                    (
                        recon_loss.item(),
                        kld_weight * local_kld.item(),
                        kld_weight * global_kld.item(),
                    )
                )

            train_loss.append(np.mean(total_loss, axis=0))
            print(
                f"[{epoch}/{epochs + 1}]; {train_loss[-1].sum()}; (recon, local, global); {train_loss[-1]})"
            )

            if epoch % max((epochs // 20), 1) == 0 or True:
                self.save_and_log(data, epoch, (niw_param, mniw_param))

        print("Finished training of the SVAE")
        self.save_and_log(data, "end", (niw_param, mniw_param))
        return train_loss
