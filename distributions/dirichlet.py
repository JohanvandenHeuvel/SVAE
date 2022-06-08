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
        alpha = self.natural_to_standard()
        value = torch.sum(torch.special.gammaln(alpha), dim=-1) - torch.special.gammaln(
            torch.sum(alpha, dim=-1)
        )
        return torch.sum(value)

    def natural_to_standard(self):
        return self.nat_param + 1

    def standard_to_natural(self, alpha):
        return alpha - 1
