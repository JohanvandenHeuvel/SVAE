import numpy as np
import torch
from torch.utils.data import DataLoader

from dense import pack_dense
from distributions import Gaussian, NormalInverseWishart, Dirichlet, Categorical
from plot import plot_latents


def initialize_global_parameters(K, D, alpha, niw_conc=10.0, random_scale=0.0):
    """

    Parameters
    ----------
    K:
        Number of clusters.
    D:
        Number of dimensions.
    alpha
    niw_conc
    random_scale

    Returns
    -------

    """

    def initialize_niw_natural_parameters(N):
        nu, S, m, kappa = (
            N + niw_conc,
            (N + niw_conc) * np.eye(N),
            np.zeros(N),
            niw_conc,
        )
        m = m + random_scale * np.random.rand(*m.shape)

        nu = torch.Tensor([nu]).unsqueeze(0)
        S = torch.Tensor(S).unsqueeze(0)
        m = torch.Tensor(m).unsqueeze(0)
        kappa = torch.Tensor([kappa]).unsqueeze(0)

        return NormalInverseWishart(None).standard_to_natural(kappa, m, S, nu)

    dirichlet_natural_parameters = alpha * (
        torch.rand(K) if random_scale else torch.ones(K)
    )
    niw_natural_parameters = torch.cat(
        [initialize_niw_natural_parameters(D) for _ in range(K)], dim=0
    )

    dirichlet_natural_parameters = dirichlet_natural_parameters.detach()
    niw_natural_parameters = niw_natural_parameters.detach()

    return dirichlet_natural_parameters, niw_natural_parameters


def initialize_meanfield(label_parameters, potentials):
    T = len(potentials)
    K = len(label_parameters)
    x = torch.rand(T, K)
    value = x / torch.sum(x, dim=-1, keepdim=True)
    return value


def prior_kld(eta_theta, eta_theta_prior):
    dir_params, niw_params = eta_theta
    dir_params_prior, niw_params_prior = eta_theta_prior

    dir = Dirichlet(dir_params)
    niw = NormalInverseWishart(niw_params)

    dir_prior = Dirichlet(dir_params_prior)
    niw_prior = NormalInverseWishart(niw_params_prior)

    expected_statistics = [dir.expected_stats(), niw.expected_stats(),][
        0
    ][0]
    difference = eta_theta[0][0] - eta_theta_prior[0][0]
    logZ_difference = (dir.logZ() + niw.logZ()) - (dir_prior.logZ() + niw_prior.logZ())

    # return torch.dot(difference, expected_statistics) - logZ_difference
    return difference * expected_statistics - logZ_difference


class SVAE:
    def __init__(self, vae):
        self.vae = vae

    def local_optimization(self, potentials, eta_theta, epochs=20):
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


        """
        priors
        """
        dir_param, niw_param = eta_theta
        label_parameters = Dirichlet(dir_param).expected_stats()
        gaussian_parameters = NormalInverseWishart(niw_param).expected_stats()

        """
        optimize local variational parameters
        """
        label_stats = initialize_meanfield(label_parameters, potentials)
        for i in range(epochs):
            """
            Gaussian x
            """
            gaussian_potentials = torch.tensordot(
                label_stats, gaussian_parameters, [[1], [0]]
            )
            eta_x = gaussian_potentials + potentials
            gaussian_stats = Gaussian(eta_x).expected_stats()

            """
            Label z
            """
            label_potentials = torch.tensordot(
                gaussian_stats, gaussian_parameters, [[1, 2], [1, 2]]
            )
            eta_z = label_potentials + label_parameters
            label_stats = Categorical(eta_z).expected_stats()

        # label_stats = label_stats.detach()
        # gaussian_potentials = torch.tensordot(
        #     label_stats, gaussian_parameters, [[1], [0]]
        # )
        # eta_x = gaussian_potentials + potentials
        # gaussian_stats = Gaussian(eta_x).expected_stats()
        #
        # label_potentials = torch.tensordot(
        #     gaussian_stats, gaussian_parameters, [[1, 2], [1, 2]]
        # )
        # eta_z = label_potentials + label_parameters
        """
        KL-Divergence
        """
        label_kld = (
            torch.tensordot(label_stats, label_potentials) - Categorical(eta_z).logZ()
        )
        gaussian_kld = (
            torch.tensordot(potentials, gaussian_stats, 3) - Gaussian(eta_x).logZ()
        )
        local_kld = label_kld + gaussian_kld

        """
        Statistics
        """
        dirichlet_stats = torch.sum(label_stats, 0)
        niw_stats = torch.tensordot(label_stats, gaussian_stats, [[0], [0]])
        prior_stats = dirichlet_stats, niw_stats

        return eta_x, eta_z, prior_stats, local_kld

    def natural_gradient(self, stats, eta_theta, eta_theta_prior, D, scale=1.0):
        """
        Natural gradient for the global variational parameters eta_theta

        Parameters
        ----------
        stats:

        eta_theta:
            Natural parameters for global variables.
        D: int
            Number of dimensions.
        scale: float
            Loss weight
        """

        def nat_grad(prior, post, s):
            return -scale / D * (prior - post + D * s)

        return (
            nat_grad(eta_theta_prior[0], eta_theta[0], stats[0]),
            nat_grad(eta_theta_prior[1], eta_theta[1], stats[1]),
        )

    def svae_objective(self, x, y, mu, log_var, global_kld, local_kld):
        """
        Monto Carlo estimate of SVAE ELBO

        Parameters
        ----------
        x: Tensor
            x_hat(phi) ~ q*(x), samples from the locally optimized q(x).
        y:

        n_batches: int
            Number of batches.
        global_kld: float

        local_kld: float

        """

        gaussian_loss = self.vae.log_likelihood(x, y, mu, log_var)

        kld_loss = global_kld - local_kld

        return gaussian_loss - kld_loss

    def fit(self, obs, K=15, batch_size=64, epochs=20):
        """
        Find the optimum for global variational parameter eta_theta, and encoder/decoder parameters.

        Parameters
        ----------
        obs:
            Observations
        epochs:
            Number of epochs to train.
        """
        _, D = obs.shape

        dataloader = DataLoader(obs, batch_size=500, shuffle=True)

        eta_theta_prior = initialize_global_parameters(
            K, D, alpha=0.05 / K, niw_conc=0.5
        )
        eta_theta = initialize_global_parameters(
            K, D, alpha=1.0, niw_conc=1.0, random_scale=3.0
        )

        vae_optimizer = torch.optim.Adam(self.vae.parameters())
        for epoch in range(epochs):

            total_loss = 0
            for i, y in enumerate(dataloader):

                # Force scale to be positive, and it's negative inverse to be negative
                mu, log_var = self.vae.encode(y.float())
                scale = -torch.exp(0.5 * log_var)
                potentials = pack_dense(scale, mu)

                """
                Find local optimum for local variational parameter eta_x, eta_z
                """
                eta_x, eta_z, prior_stats, local_kld = self.local_optimization(
                    potentials, eta_theta
                )

                x = Gaussian(eta_x).rsample()

                """
                Update global variational parameter eta_theta using natural gradient
                """
                nat_grad = self.natural_gradient(
                    prior_stats, eta_theta, eta_theta_prior, D
                )

                # TODO should add own version of SGD here
                eta_theta = tuple(
                    [eta_theta[i] - nat_grad[i] for i in range(len(eta_theta))]
                )
                # TODO fix this, should not be detached
                eta_theta = [eta_theta[i].detach() for i in range(len(eta_theta))]

                """
                Update encoder/decoder parameters using automatic differentiation
                """
                global_kld = prior_kld(eta_theta, eta_theta_prior)

                mu, log_var = self.vae.decode(x)
                vae_optimizer.zero_grad()
                loss = self.svae_objective(x, y, mu, log_var, global_kld, local_kld)
                loss.backward()
                vae_optimizer.step()

                total_loss += loss

            print(f"Epoch:{epoch}/{epochs} [loss: {total_loss:.3f}]")
            if epoch % 1 == 0:
                plot_latents(x, eta_theta)
