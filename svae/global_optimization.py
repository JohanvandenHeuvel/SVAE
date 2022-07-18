from typing import Tuple

import numpy as np
import torch

from distributions import (
    NormalInverseWishart,
    MatrixNormalInverseWishart,
    Dirichlet,
    exponential_kld,
)


def initialize_global_lds_parameters(n, scale=1.0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nu = torch.tensor([n + 1])
    Phi = 2 * scale * (n + 1) * torch.eye(n)
    mu_0 = torch.zeros(n)
    kappa = torch.tensor([1 / (2 * scale * n)])

    M = torch.eye(n)
    K = 1 / (2 * scale * n) * torch.eye(n)

    init_state_prior = NormalInverseWishart(torch.zeros_like(nu)).standard_to_natural(
        kappa.unsqueeze(0), mu_0.unsqueeze(0), Phi.unsqueeze(0), nu.unsqueeze(0)
    )
    dynamics_prior = MatrixNormalInverseWishart(
        torch.zeros_like(nu)
    ).standard_to_natural(nu, Phi, M, K)

    dynamics_prior = tuple([d.to(device) for d in dynamics_prior])

    return init_state_prior.to(device), dynamics_prior


def initialize_global_gmm_parameters(
    K: int, D: int, alpha: float, niw_conc: float, random_scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize the parameters for the Dirichlet distribution (labels) and
    the NormalInverseWishart distribution (clusters).

    Parameters
    ----------
    K:
        Number of clusters.
    D:
        Dimension of the data.
    alpha:
        Prior for the Dirichlet distribution.
    niw_conc:
        NormalInverseWishart concentration.
    random_scale:
        Scale of the means of the clusters.

    Returns
    -------

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def initialize_niw_natural_parameters(D: int) -> torch.Tensor:
        nu, Phi, m_0, kappa = (
            D + niw_conc,
            (D + niw_conc) * np.eye(D),
            np.zeros(D),
            niw_conc,
        )
        m_0 = m_0 + random_scale * np.random.randn(*m_0.shape)

        nu = torch.Tensor([nu]).unsqueeze(0)
        Phi = torch.Tensor(Phi).unsqueeze(0)
        m_0 = torch.Tensor(m_0).unsqueeze(0)
        kappa = torch.Tensor([kappa]).unsqueeze(0)

        nat_param = NormalInverseWishart(torch.zeros_like(nu)).standard_to_natural(
            kappa, m_0, Phi, nu
        )
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


def natural_gradient(
    stats: Tuple[torch.Tensor, torch.Tensor],
    eta_theta: Tuple[torch.Tensor, torch.Tensor],
    eta_theta_prior: Tuple[torch.Tensor, torch.Tensor],
    N: int,
    num_batches: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Natural gradient for the global variational parameters eta_theta

    Parameters
    ----------
    stats:
       Sufficient statistics.
    eta_theta:
        Posterior natural parameters for global variables.
    eta_theta_prior:
        Prior natural parameters for global variables.
    N:
        Number of data-points.
    num_batches:
        Number of batches in the data.
    """

    def nat_grad(prior, posterior, s) -> torch.Tensor:
        if isinstance(prior, Tuple):
            return [
                -1.0 / N * prior[j] - posterior[j] + num_batches * s[j]
                for j in range(len(prior))
            ]
        return -1.0 / N * (prior - posterior + num_batches * s)

    value = [
        nat_grad(eta_theta_prior[i], eta_theta[i], stats[i])
        for i in range(len(eta_theta))
    ]
    return value


def prior_kld_lds(eta_theta, eta_theta_prior, eta_x):
    niw_params, mniw_params = eta_theta
    niw_params_prior, mniw_params_prior = eta_theta_prior

    niw = NormalInverseWishart(niw_params)
    niw_prior = NormalInverseWishart(niw_params_prior)
    # niw_kld = exponential_kld(niw, niw_prior, eta_x[0])
    niw_kld = exponential_kld(niw, niw_prior)

    mniw = MatrixNormalInverseWishart(mniw_params)
    mniw_prior = MatrixNormalInverseWishart(mniw_params_prior)
    # mniw_kld = exponential_kld(mniw, mniw_prior, eta_x[1])
    mniw_kld = exponential_kld(mniw, mniw_prior)

    return mniw_kld + niw_kld


def prior_kld_gmm(
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
