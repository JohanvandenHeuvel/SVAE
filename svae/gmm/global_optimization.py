from typing import Tuple

import numpy as np
import torch

from distributions import (
    NormalInverseWishart,
    Dirichlet,
    exponential_kld,
)


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

    return dirichlet_natural_parameters.to(device).double(), niw_natural_parameters.to(device).double()


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
