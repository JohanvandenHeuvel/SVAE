import torch

from .distribution import ExpDistribution

from dense import pack_dense, unpack_dense


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

        assert scale.ndim == 3

        E_x = loc
        E_xxT = scale + torch.einsum("bi,bj->bij", (E_x, E_x))
        E_n = torch.ones(len(scale))

        return pack_dense(E_xxT, E_x, E_n, E_n)

    def logZ(self):
        loc, scale = self.natural_to_standard()
        value = (
            -1
            / 2
            * (
                torch.slogdet(scale)[1]
                + torch.bmm(loc.unsqueeze(1), torch.bmm(torch.inverse(scale), loc[..., None])).squeeze()
            )
        )
        return value

    def natural_to_standard(self):
        eta_2, eta_1, _, _ = unpack_dense(self.nat_param)

        scale = -1 / 2 * torch.inverse(eta_2)
        loc = torch.bmm(scale, eta_1[..., None]).squeeze()

        return loc, scale

    def standard_to_natural(self, loc, scale):
        eta_1 = torch.inverse(scale) @ loc
        eta_2 = torch.flatten(-1 / 2 * torch.inverse(scale))

        return pack_dense(eta_2, eta_1)

    def rsample(self):
        loc, scale = self.natural_to_standard()
        eps = torch.randn_like(loc)

        return loc + torch.matmul(scale, torch.ones(loc.shape[1])) * eps

