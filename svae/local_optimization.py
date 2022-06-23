from typing import Tuple

import numpy as np
import torch

from distributions import (
    Gaussian,
    NormalInverseWishart,
    Dirichlet,
    Categorical,
)


def initialize_meanfield(
    label_parameters: torch.Tensor, potentials: torch.Tensor
) -> torch.Tensor:
    device = potentials.device
    T = len(potentials)
    K = len(label_parameters)
    x = torch.rand(T, K, device=device)
    value = x / torch.sum(x, dim=-1, keepdim=True)
    return value


def local_optimization(
    potentials: torch.Tensor,
    eta_theta: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], float]:
    """
    Find the optimum for local variational parameters eta_x, eta_z

    Parameters
    ----------
    potentials:
        Output of the encoder network.
    eta_theta:
        Natural global parameters for Q(theta).
    epochs:
        Number of epochs to train.
    """

    def gaussian_optimization(
        gaussian_parameters: torch.Tensor,
        potentials: torch.Tensor,
        label_stats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        gaussian_potentials = torch.tensordot(
            label_stats, gaussian_parameters, [[1], [0]]
        )
        eta_x = gaussian_potentials + potentials
        gaussian_stats = Gaussian(eta_x).expected_stats()
        gaussian_kld = (
            torch.tensordot(potentials, gaussian_stats, 3) - Gaussian(eta_x).logZ()
        )
        return eta_x, gaussian_stats, gaussian_kld.item()

    def label_optimization(
        gaussian_parameters: torch.Tensor,
        label_parameters: torch.Tensor,
        gaussian_stats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        label_potentials = torch.tensordot(
            gaussian_stats, gaussian_parameters, [[1, 2], [1, 2]]
        )
        eta_z = label_potentials + label_parameters
        label_stats = Categorical(eta_z).expected_stats()
        label_kld = (
            torch.tensordot(label_stats, label_potentials) - Categorical(eta_z).logZ()
        )
        return eta_z, label_stats, label_kld.item()

    # with torch.no_grad():
    """
    priors
    """
    dir_param, niw_param = eta_theta
    label_parameters = Dirichlet(dir_param).expected_stats()
    gaussian_parameters = NormalInverseWishart(niw_param).expected_stats()

    # with torch.no_grad():
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
    _, label_stats, label_kld = label_optimization(
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
