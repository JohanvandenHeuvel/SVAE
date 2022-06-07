import torch

from .distribution import ExpDistribution


class Gaussian(ExpDistribution):
    def __init__(self, nat_param):
        super().__init__(nat_param)

    def expected_stats(self):
        return 0

    def logZ(self):
        pass

    def natural_to_standard(self):
        eta_1, eta_2 = self.nat_param

        loc = -1 / 2 * torch.inverse(eta_2) @ eta_1
        scale = -1 / 2 * torch.inverse(eta_2)

        return loc, scale

    def standard_to_natural(self, loc, scale):
        eta_1 = torch.inverse(scale) @ loc
        eta_2 = torch.flatten(-1 / 2 * torch.inverse(scale))

        return eta_1, eta_2
