from abc import ABC

import torch


class Dirichlet(torch.distributions.Dirichlet):
    def __init__(self, concentration):
        super().__init__(concentration)

    def expected_stats(self):
        """
        E[log X]
        """
        stats = torch.digamma(self.concentration) - torch.digamma(
            self.concentration.sum()
        )
        return stats
