import torch

from .distribution import ExpDistribution


class Dirichlet(ExpDistribution):
    def __init__(self, nat_param):
        super().__init__(nat_param)

    def expected_stats(self):
        alpha = self.natural_to_standard()
        stats = torch.digamma(alpha) - torch.digamma(alpha.sum())
        return stats

    def logZ(self):
        return 1

    def natural_to_standard(self):
        return self.nat_param + 1

    def standard_to_natural(self, alpha):
        return alpha - 1
