from abc import ABC, abstractmethod
from typing import Tuple

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

    @abstractmethod
    def standard_to_natural(self, *args) -> torch.Tensor:
        """Convert standard parameters to natural parameters."""
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


def exponential_kld(dist_1: ExpDistribution, dist_2: ExpDistribution, expected_stats=None) -> float:
    if expected_stats is None:
        expected_stats = dist_1.expected_stats()
    # TODO sometimes gives negative values
    value = (
        torch.flatten((dist_1.nat_param - dist_2.nat_param))
        @ torch.flatten(expected_stats)
        - dist_1.logZ()
        + dist_2.logZ()
    )
    assert value >= 0
    return value.item()
