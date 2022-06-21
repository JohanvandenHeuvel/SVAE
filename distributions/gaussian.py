import torch

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

        E_x = loc
        E_xxT = scale + torch.einsum("bi,bj->bij", (E_x, E_x))
        E_n = torch.ones(len(scale), device=self.device)

        return pack_dense(E_xxT, E_x, E_n, E_n)

    def logZ(self):
        """normalization constant
        """

        # USING STANDARD PARAMETERS (easier to understand)
        # loc, scale = self.natural_to_standard()
        """
        We can do the following which is much more computationally efficient
            log det (scale) = 2 sum log diag (L)
        where L is the lower triangular matrix produced by Cholesky decomposition of scale (psd) matrix.
        """
        # L = torch.linalg.cholesky(scale)
        # value = (
        #     2 * torch.sum(torch.log(torch.diagonal(L, dim1=-1, dim2=-2)), dim=-1)
        #     + torch.bmm(
        #         loc.unsqueeze(1), torch.bmm(torch.inverse(scale), loc[..., None])
        #     ).squeeze()
        #     + 2 * (a + b)
        # )
        # return 1 / 2 * torch.sum(value)

        # USING NATURAL PARAMETERS (faster)
        # TODO I don't understand why there is a (a+b) in the normalization constant.
        eta_2, eta_1, a, b = unpack_dense(self.nat_param)

        L = torch.linalg.cholesky(-2 * eta_2)
        value = (
            1 / 2 * torch.sum(eta_1 * torch.linalg.solve(-2 * eta_2, eta_1))
            - torch.sum(torch.log(torch.diagonal(L, dim1=-1, dim2=-2)))
            + torch.sum(a + b)
        )

        return value

    def natural_to_standard(self):
        eta_2, eta_1, _, _ = unpack_dense(self.nat_param)

        scale = -1 / 2 * torch.inverse(eta_2)
        loc = torch.bmm(scale, eta_1[..., None]).squeeze()

        return loc, scale

    def standard_to_natural(self, loc, scale):
        scale_inv = torch.inverse(scale)
        eta_1 = scale_inv @ loc
        eta_2 = torch.flatten(-1 / 2 * scale_inv)

        return pack_dense(eta_2, eta_1)

    def rsample(self):
        loc, scale = self.natural_to_standard()
        eps = torch.randn_like(loc)
        samples = (
            loc
            + torch.matmul(scale, torch.ones(loc.shape[1], device=self.device)) * eps
        )

        return samples
