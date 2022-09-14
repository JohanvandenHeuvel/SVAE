import random

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from matrix_ops import pack_dense, unpack_dense
from seed import SEED
from .distribution import ExpDistribution

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def info_to_natural(J, h):
    eta_2 = -0.5 * J
    eta_1 = h
    return pack_dense(eta_2.unsqueeze(0), eta_1.unsqueeze(0))


def natural_to_info(nat_param):
    eta_2, eta_1, _, _ = unpack_dense(nat_param)
    J = -2 * eta_2
    h = eta_1
    return J, h


def info_to_standard(J, h):
    J_inv = torch.inverse(J)
    scale = J_inv
    loc = J_inv @ h
    return loc, scale


class Gaussian(ExpDistribution):
    def __init__(self, nat_param: torch.Tensor):
        super().__init__(nat_param)
        self.device = nat_param.device

    def expected_stats(self) -> torch.Tensor:
        """Compute the expected statistics of the multivariate Gaussian.

        Returns
        -------
        E_x:
            Expected value of x
        E_xxT:
            Expected value of xxT
        E_n:
            Identity.
        """
        loc, scale = self.natural_to_standard()

        E_x = loc
        E_xxT = scale + torch.einsum("bi,bj->bij", (E_x, E_x))
        E_n = torch.ones(len(scale), device=self.device)

        return pack_dense(E_xxT, E_x, E_n, E_n)

    def logZ(self):
        def using_standard_parameters():
            loc, scale = self.natural_to_standard()
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

        def using_natural_parameters():
            # TODO I don't understand why there is a (a+b) in the normalization constant.
            eta_2, eta_1, a, b = unpack_dense(self.nat_param)

            L = torch.linalg.cholesky(-2 * eta_2)
            value = (
                1 / 2 * torch.sum(eta_1 * torch.linalg.solve(-2 * eta_2, eta_1))
                - torch.sum(torch.log(torch.diagonal(L, dim1=-1, dim2=-2)))
                + torch.sum(a + b)
            )

            return value

        return using_natural_parameters()

    def natural_to_standard(self):
        eta_2, eta_1, _, _ = unpack_dense(self.nat_param)

        L = torch.linalg.cholesky(-2 * eta_2)
        # scale = -1 / 2 * torch.inverse(eta_2)
        scale = torch.cholesky_inverse(L)
        loc = torch.bmm(scale, eta_1[..., None])
        return loc.squeeze(), scale.squeeze()

    def rsample(self, n_samples=1):
        """get samples using the re-parameterization trick and natural parameters"""
        loc, scale = self.natural_to_standard()
        return MultivariateNormal(loc, scale).rsample([n_samples])


def standard_pair_params(J11, J12, J22):
    Q = torch.inverse(J22)
    A = -(J12 @ Q).T
    return A, Q
