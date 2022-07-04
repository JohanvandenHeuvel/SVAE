from typing import Tuple

import numpy as np
import torch

from distributions import NormalInverseWishart


def initialize_global_lds_parameters(n, scale=1.0):

    nu = n + 1
    Phi = 2 * scale * (n + 1) * np.eye(n)
    mu_0 = np.zeros(n)
    kappa = 1 / (2 * scale * n)

    M = np.eye(n)
    K = 1 / (2 * scale * n) * np.eye(n)

    init_state_prior = niw.standard_to_natural(nu, Phi, mu_0, kappa)
    dynamics_prior = mniw.standard_to_natural(nu, Phi, M, K)

    return init_state_prior, dynamics_prior


def initialize_global_parameters(
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
        return -1.0 / N * (prior - posterior + num_batches * s)

    value = (
        nat_grad(eta_theta_prior[0], eta_theta[0], stats[0]),
        nat_grad(eta_theta_prior[1], eta_theta[1], stats[1]),
    )
    return value
