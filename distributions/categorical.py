import torch
import torch.nn.functional as F

from .distribution import ExpDistribution

from scipy.stats import multinomial


def sample(p, n=1):
    """

    Parameters
    ----------
    p:
        Probability for each category, should sum to 1.0.
        If multiple arrays are given, this is repeated for every array.
    n:
        number of samples

    Returns
    -------

    """
    if len(p.shape) == 2:
        return [multinomial.rvs(n, _p) for _p in p]
    return multinomial.rvs(n, p)


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
