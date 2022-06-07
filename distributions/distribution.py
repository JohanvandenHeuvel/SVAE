from abc import ABC, abstractmethod


class ExpDistribution(ABC):
    def __init__(self, nat_param):
        self._nat_param = nat_param

    @abstractmethod
    def expected_stats(self):
        """Compute expected statistics."""
        pass

    @abstractmethod
    def logZ(self):
        """Compute log normalization constant."""
        pass

    @abstractmethod
    def natural_to_standard(self):
        """Convert natural parameters to standard parameters."""
        pass

    @abstractmethod
    def standard_to_natural(self, *args):
        """Convert standard parameters to natural parameters."""
        pass

    @property
    def nat_param(self):
        """Get the natural parameters.

        Returns
        -------
        List
            List of the natural parameters.
        """
        return self._nat_param

    @nat_param.setter
    def nat_param(self, value):
        """Set the natural parameters.

        Parameters
        ----------
        value : List
            List of the natural parameters.
        """
        self._nat_param = value
