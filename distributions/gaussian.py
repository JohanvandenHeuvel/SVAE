import torch
import numpy as np

from dense import pack_dense, unpack_dense
from .distribution import ExpDistribution


class Gaussian(ExpDistribution):
    def __init__(self, nat_param):
        super().__init__(nat_param)
        self.device = nat_param.device

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
        E_n = torch.ones(len(scale), device=self.device)

        return pack_dense(E_xxT, E_x, E_n, E_n)

    def logZ(self):
        loc, scale = self.natural_to_standard()

        # TODO I don't understand why there is a (a+b) in the normalization constant.
        _, _, a, b = unpack_dense(self.nat_param)

        """
        We can do the following which is much more computationally efficient
            log det (scale) = 2 sum log diag (L)
        where L is the lower triangular matrix produced by Cholesky decomposition of scale (psd) matrix.
        """
        L = torch.linalg.cholesky(scale)
        value = (
            2 * torch.sum(torch.log(torch.diagonal(L, dim1=-1, dim2=-2)), dim=-1)
            + torch.bmm(
                loc.unsqueeze(1), torch.bmm(torch.inverse(scale), loc[..., None])
            ).squeeze()
            + 2 * (a + b)
        )

        return 1 / 2 * torch.sum(value)

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
        samples = (
            loc
            + torch.matmul(scale, torch.ones(loc.shape[1], device=self.device)) * eps
        )

        return samples
