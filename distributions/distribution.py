from abc import ABC, abstractmethod
from typing import Tuple, List

import torch
import numpy as np


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


def exponential_kld(dist_1: ExpDistribution, dist_2: ExpDistribution, expected_stats=None) -> float:
    if expected_stats is None:
        expected_stats = dist_1.expected_stats()
    # TODO sometimes gives negative values
    P = dist_1.nat_param
    Q = dist_2.nat_param
    if isinstance(P, List):
        P = np.array([x.cpu().detach().numpy() for x in P])
        Q = np.array([x.cpu().detach().numpy() for x in Q])
        expected_stats = np.array([x.cpu().detach().numpy() for x in expected_stats])
        value = (
                (P - Q).flatten()
                @ expected_stats.flatten()
                - dist_1.logZ()
                + dist_2.logZ()
        )
    else:
        value = (
            torch.flatten((P - Q))
            @ torch.flatten(expected_stats)
            - dist_1.logZ()
            + dist_2.logZ()
        )
    assert value >= 0
    return value.item()
