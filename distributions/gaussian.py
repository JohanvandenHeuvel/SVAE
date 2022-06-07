import torch

from .distribution import ExpDistribution


class Gaussian(ExpDistribution):
    def __init__(self, nat_param):
        super().__init__(nat_param)

    def expected_stats(self):
        """Compute the expected statistics of the multivariate Gaussian.

        Returns
        -------
        E_x : torch.Tensor
            Expected value of x
        E_xxT : torch.Tensor
            Expected value of xxT
        """
        loc, scale = self.natural_to_standard()

        E_x = loc
        E_xxT = scale + torch.outer(E_x, E_x)

        return E_x, E_xxT

    def logZ(self):
        loc, scale = self.natural_to_standard()
        value = (
            -1 / 2 * (torch.log(torch.det(scale)) + loc.T @ torch.inverse(scale) @ loc)
        )
        return value

    def natural_to_standard(self):
        eta_1, eta_2 = self.nat_param

        loc = -1 / 2 * torch.inverse(eta_2) @ eta_1
        scale = -1 / 2 * torch.inverse(eta_2)

        return loc, scale

    def standard_to_natural(self, loc, scale):
        eta_1 = torch.inverse(scale) @ loc
        eta_2 = torch.flatten(-1 / 2 * torch.inverse(scale))

        return eta_1, eta_2
