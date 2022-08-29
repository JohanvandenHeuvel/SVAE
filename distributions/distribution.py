from abc import ABC, abstractmethod
from typing import Tuple, List

import torch


class ExpDistribution(ABC):
    def __init__(self, nat_param: torch.Tensor):
        self._nat_param = nat_param

    @abstractmethod
    def expected_stats(self) -> torch.Tensor:
        """Compute expected statistics."""
        pass

    @abstractmethod
    def logZ(self) -> float:
        """Compute log normalization constant."""
        pass

    @abstractmethod
    def natural_to_standard(self) -> Tuple:
        """Convert natural parameters to standard parameters."""
        pass

    @property
    def nat_param(self) -> torch.Tensor:
        """Get the natural parameters.

        Returns
        -------
        List
            List of the natural parameters.
        """
        return self._nat_param

    @nat_param.setter
    def nat_param(self, value: torch.Tensor):
        """Set the natural parameters.

        Parameters
        ----------
        value : List
            List of the natural parameters.
        """
        self._nat_param = value


def exponential_kld(dist_1: ExpDistribution, dist_2: ExpDistribution) -> float:
    # TODO sometimes gives negative values
    expected_stats = dist_1.expected_stats()
    eta_1 = dist_1.nat_param
    eta_2 = dist_2.nat_param
    if isinstance(eta_1, List):
        # for i in range(4):
        #     print((eta_1[i] - eta_2[i]).cpu().detach().numpy())
        value = [torch.flatten(eta_1[i] - eta_2[i]).float() @ torch.flatten(expected_stats[i].float()) for i in range(len(eta_1))]
        value = torch.sum(torch.stack(value))
        value = value - (dist_1.logZ() - dist_2.logZ())
        # print(f"KLD: {value:.3f} {dist_1.logZ():.3f}")
        value = value
    else:
        value = (
            torch.flatten((eta_1 - eta_2))
            @ torch.flatten(expected_stats)
            - dist_1.logZ()
            + dist_2.logZ()
        )
        value = value
    return value
