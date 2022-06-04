import torch
from torch import nn

import numpy as np

from distributions import Gaussian, NormalInverseWishart, Dirichlet, Categorical

from label import Label


def initialize_global_parameters(K, N, alpha, niw_conc=10.0, random_scale=0.0):
    def initialize_niw_natural_parameters(N):
        nu, S, m, kappa = (
            N + niw_conc,
            (N + niw_conc) * np.eye(N),
            np.zeros(N),
            niw_conc,
        )
        m = m + random_scale * np.random.rand(*m.shape)
        return NormalInverseWishart().standard_to_natural(S, m, kappa, nu)

    dirichlet_natural_parameters = alpha * (
        np.random.rand(K) if random_scale else np.ones(K)
    )
    niw_natural_parameters = np.stack(
        [initialize_niw_natural_parameters(N) for _ in range(K)]
    )

    return dirichlet_natural_parameters, niw_natural_parameters


def prior_kld(eta_theta, eta_theta_prior):

    dirichlet_natparams, niw_natparams = eta_theta
    dirichlet_natparams_prior, niw_natparams_prior = eta_theta_prior

    expected_statistics = [
        Dirichlet.expected_stats(dirichlet_natparams),
        NormalInverseWishart.expected_stats(niw_natparams),
    ]
    difference = eta_theta - eta_theta_prior
    logZ_difference = (
        Dirichlet.logZ(dirichlet_natparams) + NormalInverseWishart.logZ(niw_natparams)
    ) - (
        Dirichlet.logZ(dirichlet_natparams_prior)
        + NormalInverseWishart.logZ(niw_natparams_prior)
    )

    return torch.dot(difference, expected_statistics) - logZ_difference


class SVAE:
    def __init__(self, vae):
        self.vae = vae

    def local_optimization(self, y, eta_theta, epochs=20):
        """
        Find the optimum for local variational parameters eta_x, eta_z

        Parameters
        ----------
        y: Tensor
            Observations
        eta_theta: Tensor
            Natural parameters for Q(theta)
        epochs: int
            Number of epochs to train.
        """
        potentials = self.vae.encode(y)

        """
        priors
        """
        label_parameters = Dirichlet.expected_stats(eta_theta)
        gaussian_parameters = NormalInverseWishart.expected_stats(eta_theta)

        """
        optimize local variational parameters
        """
        eta_z = 0
        eta_x = 0

        label_stats = Label.expected_stats(eta_z)
        for i in range(epochs):

            """
            Gaussian x
            """
            gaussian_potentials = torch.tensordot(
                label_stats, gaussian_parameters, [[1], [0]]
            )
            eta_x = gaussian_potentials + potentials
            gaussian_stats = Gaussian.expected_stats(eta_x)

            """
            Label z
            """
            label_potentials = torch.tensordot(
                gaussian_stats, gaussian_parameters, [[1, 2], [1, 2]]
            )
            eta_z = label_potentials + label_parameters
            label_stats = Label.expected_stats(eta_z)

        """
        KL-Divergence
        """
        label_kld = torch.tensordot(label_stats, label_potentials) - Categorical.logZ(
            eta_z
        )
        gaussian_kld = torch.tensordot(potentials, gaussian_stats, 3) - Gaussian.logZ(
            eta_x
        )
        local_kld = label_kld + gaussian_kld

        """
        Statistics
        """
        dirichlet_stats = torch.sum(label_stats, 0)
        niw_stats = torch.tensordot(label_stats, gaussian_stats, [[0], [0]])
        prior_stats = dirichlet_stats, niw_stats

        return eta_x, eta_z, prior_stats, local_kld

    def natural_gradient(self, stats, eta_theta, eta_theta_prior, N, scale=1.0):
        """
        Natural gradient for the global variational parameters eta_theta

        Parameters
        ----------
        stats:

        eta_theta:
            Natural parameters for global variables.
        N: int
            Number of data-points.
        scale: float
            Loss weight
        """
        return -scale / N * (eta_theta_prior - eta_theta + N * stats)

    def svae_objective(self, x, n_batches, global_kld, local_kld):
        """
        Monto Carlo estimate of SVAE ELBO

        Parameters
        ----------
        x: Tensor
            x_hat(phi) ~ q*(x), samples from the locally optimized q(x).
        n_batches: int
            Number of batches.
        global_kld: float

        local_kld: float

        """

        gaussian_loss = n_batches * self.vae.log_likelihood(x, get_batch(i))

        kld_loss = global_kld - n_batches * local_kld

        return gaussian_loss - kld_loss


def train(self, y, epochs):
    """
    Find the optimum for global variational parameter eta_theta, and encoder/decoder parameters.

    Parameters
    ----------
    y:
        Observations
    epochs:
        Number of epochs to train.
    """
    N, K = y.shape

    eta_theta_prior = initialize_global_parameters(K, N, alpha=0.05 / K, niw_conc=0.5)
    eta_theta = initialize_global_parameters(
        K, N, alpha=1.0, niw_conc=1.0, random_scale=3.0
    )

    vae_optimizer = torch.optim.SGD(self.vae, 0.1)
    for epoch in range(epochs):

        """
        Find local optimum for local variational parameter eta_x
        """
        eta_x, eta_z, prior_stats, local_kld = self.local_optimization(y, eta_theta)

        # TODO use re-parameterization for this sampling
        x = Gaussian(eta_x).rsample()

        """
        Update global variational parameter eta_theta using natural gradient
        """
        nat_grad = self.natural_gradient(prior_stats, eta_theta, eta_theta_prior, N)

        # TODO should add own version of SGD here
        eta_theta -= nat_grad

        """
        Update encoder/decoder parameters using automatic differentiation
        """
        global_kld = prior_kld(eta_theta, eta_theta_prior)

        vae_optimizer.zero_grad()
        loss = self.svae_objective(x, n_batches, global_kld, local_kld)
        loss.backward()
        vae_optimizer.step()
