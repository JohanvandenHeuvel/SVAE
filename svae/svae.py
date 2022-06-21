import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dense import pack_dense, unpack_dense
from distributions import (
    Gaussian,
    NormalInverseWishart,
    Dirichlet,
    Categorical,
    exponential_kld,
)
from plot.plot import plot_reconstruction


def initialize_global_parameters(K, D, alpha, niw_conc, random_scale):
    """

    Parameters
    ----------
    K:
        Number of clusters.
    D:
        Number of dimensions.
    alpha:

    niw_conc:

    random_scale:


    Returns
    -------

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def initialize_niw_natural_parameters(D):
        nu, S, m, kappa = (
            D + niw_conc,
            (D + niw_conc) * np.eye(D),
            np.zeros(D),
            niw_conc,
        )
        m = m + random_scale * np.random.randn(*m.shape)

        nu = torch.Tensor([nu]).unsqueeze(0)
        S = torch.Tensor(S).unsqueeze(0)
        m = torch.Tensor(m).unsqueeze(0)
        kappa = torch.Tensor([kappa]).unsqueeze(0)

        nat_param = NormalInverseWishart(torch.zeros_like(nu)).standard_to_natural(kappa, m, S, nu)
        return nat_param

    dirichlet_natural_parameters = alpha * (
        torch.rand(K) if random_scale else torch.ones(K)
    )
    niw_natural_parameters = torch.cat(
        [initialize_niw_natural_parameters(D) for _ in range(K)], dim=0
    )

    dirichlet_natural_parameters = dirichlet_natural_parameters.detach()
    niw_natural_parameters = niw_natural_parameters.detach()

    return dirichlet_natural_parameters.to(device), niw_natural_parameters.to(device)


def initialize_meanfield(label_parameters, potentials):
    device = potentials.device
    T = len(potentials)
    K = len(label_parameters)
    x = torch.rand(T, K, device=device)
    value = x / torch.sum(x, dim=-1, keepdim=True)
    return value


def prior_kld(eta_theta, eta_theta_prior):
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
    def __init__(self, vae):
        self.vae = vae
        self.device = vae.device

    def local_optimization(self, potentials, eta_theta, epochs=100):
        """
        Find the optimum for local variational parameters eta_x, eta_z

        Parameters
        ----------
        potentials: Tensor

        eta_theta: Tensor
            Natural parameters for Q(theta)
        epochs: int
            Number of epochs to train.
        """

        def gaussian_optimization(gaussian_parameters, potentials, label_stats):
            gaussian_potentials = torch.tensordot(
                label_stats, gaussian_parameters, [[1], [0]]
            )
            eta_x = gaussian_potentials + potentials
            gaussian_stats = Gaussian(eta_x).expected_stats()
            gaussian_kld = (
                torch.tensordot(potentials, gaussian_stats, 3) - Gaussian(eta_x).logZ()
            )
            return eta_x, gaussian_stats, gaussian_kld

        def label_optimization(gaussian_parameters, label_parameters, gaussian_stats):
            label_potentials = torch.tensordot(
                gaussian_stats, gaussian_parameters, [[1, 2], [1, 2]]
            )
            eta_z = label_potentials + label_parameters
            label_stats = Categorical(eta_z).expected_stats()
            label_kld = (
                torch.tensordot(label_stats, label_potentials)
                - Categorical(eta_z).logZ()
            )
            return eta_z, label_stats, label_kld

        """
        priors
        """
        dir_param, niw_param = eta_theta
        label_parameters = Dirichlet(dir_param.detach()).expected_stats()
        gaussian_parameters = NormalInverseWishart(niw_param.detach()).expected_stats()

        """
        optimize local variational parameters
        """
        kl = np.inf
        label_stats = initialize_meanfield(label_parameters, potentials)
        for i in range(epochs):
            """
            Gaussian x
            """
            _, gaussian_stats, gaussian_kld = gaussian_optimization(
                gaussian_parameters, potentials, label_stats
            )

            """
            Label z
            """
            _, label_stats, label_kld = label_optimization(
                gaussian_parameters, label_parameters, gaussian_stats
            )

            # early stopping
            prev_l = kl
            kl = label_kld + gaussian_kld
            if abs(kl - prev_l) < 1e-3:
                break
        else:
            print("iteration limit reached")

        eta_x, gaussian_stats, gaussian_kld = gaussian_optimization(
            gaussian_parameters, potentials, label_stats
        )
        eta_z, label_stats, label_kld = label_optimization(
            gaussian_parameters, label_parameters, gaussian_stats
        )

        """
        KL-Divergence
        """
        local_kld = label_kld + gaussian_kld

        """
        Statistics
        """
        dirichlet_stats = torch.sum(label_stats, 0)
        niw_stats = torch.tensordot(label_stats, gaussian_stats, [[0], [0]])
        prior_stats = dirichlet_stats, niw_stats

        return eta_x, label_stats, prior_stats, local_kld

    def natural_gradient(
        self, stats, eta_theta, eta_theta_prior, N, num_batches, scale=10000.0
    ):
        """
        Natural gradient for the global variational parameters eta_theta

        Parameters
        ----------
        stats:

        eta_theta:
            Natural parameters for global variables.
        N: int
            Number of datapoints.
        scale: float
            Loss weight
        """

        def nat_grad(prior, post, s):
            return -scale / N * (prior - post + num_batches * s)

        value = (
            nat_grad(eta_theta_prior[0], eta_theta[0], stats[0]),
            nat_grad(eta_theta_prior[1], eta_theta[1], stats[1]),
        )
        return value

    def save_and_log(self, obs, epoch, save_path, eta_theta):
        data = torch.tensor(obs).to(self.vae.device).float()
        mu, log_var = self.vae.encode(data)
        scale = -torch.exp(0.5 * log_var)
        potentials = pack_dense(scale, mu)

        eta_x, label_stats, _, _ = self.local_optimization(potentials, eta_theta)

        # get encoded means
        gaussian_stats = Gaussian(eta_x).expected_stats()
        _, Ex, _, _ = unpack_dense(gaussian_stats)

        # get reconstructions
        x = Gaussian(eta_x).rsample()
        mu_y, log_var_y = self.vae.decode(x)

        plot_reconstruction(
            obs,
            mu_y.cpu().detach().numpy(),
            log_var_y.cpu().detach().numpy(),
            Ex.cpu().detach().numpy(),
            eta_theta,
            classes=torch.argmax(label_stats, dim=-1).cpu().detach().numpy(),
            title=f"{epoch}_svae_recon",
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
        """
        print("Training the SVAE ...")
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

        optimizer = torch.optim.Adam(self.vae.parameters())

        train_loss = []
        self.save_and_log(obs, "pre", save_path, eta_theta)
        for epoch in tqdm(range(epochs)):

            total_loss = []
            for i, y in enumerate(dataloader):
                y = y.float()
                # Force scale to be positive, and it's negative inverse to be negative
                mu, log_var = self.vae.encode(y)
                scale = -torch.exp(0.5 * log_var)
                potentials = pack_dense(scale, mu)

                """
                Find local optimum for local variational parameter eta_x, eta_z
                """
                eta_x, _, prior_stats, local_kld = self.local_optimization(
                    potentials, eta_theta
                )

                x = Gaussian(eta_x).rsample()

                """
                Update global variational parameter eta_theta using natural gradient
                """
                nat_grad = self.natural_gradient(
                    prior_stats, eta_theta, eta_theta_prior, len(obs), num_batches
                )

                step_size = 1e-3
                eta_theta = tuple(
                    [
                        eta_theta[i] - step_size * nat_grad[i]
                        for i in range(len(eta_theta))
                    ]
                )

                """
                Update encoder/decoder parameters using automatic differentiation
                """
                global_kld = prior_kld(eta_theta, eta_theta_prior)

                mu_y, log_var_y = self.vae.decode(x)
                recon_loss = self.vae.loss_function(y, mu_y, log_var_y)
                kld_loss = local_kld
                loss = recon_loss + kld_weight * kld_loss

                optimizer.zero_grad()
                # compute gradients
                loss.backward()
                # update parameters
                optimizer.step()

                total_loss.append((recon_loss.item(), kld_weight * kld_loss.item()))
            train_loss.append(np.mean(total_loss, axis=0))

            if epoch % max((epochs // 10), 1) == 0:
                self.save_and_log(obs, epoch, save_path, eta_theta)

        print("Finished training of the SVAE")
        return train_loss
