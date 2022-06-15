import torch
import torch.nn.functional as F

from .distribution import ExpDistribution


class Categorical(ExpDistribution):

    def __init__(self, nat_param):
        super().__init__(nat_param)

    def expected_stats(self):
        return F.softmax(self.nat_param, dim=-1)

    def logZ(self):
        return torch.sum(torch.logsumexp(self.nat_param, dim=-1))

    def natural_to_standard(self):
        pass

    def standard_to_natural(self, *args):
        pass
