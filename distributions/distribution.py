from abc import ABC, abstractmethod


class ExpDistribution(ABC):

    def __init__(self, nat_param):
        self._nat_param = nat_param

    @abstractmethod
    def expected_stats(self):
        pass

    @abstractmethod
    def logZ(self):
        pass

    @abstractmethod
    def natural_to_standard(self):
        pass

    @abstractmethod
    def standard_to_natural(self, *args):
        pass

    @property
    def nat_param(self):
        return self._nat_param

    @nat_param.setter
    def nat_param(self, value):
        self._nat_param = value
