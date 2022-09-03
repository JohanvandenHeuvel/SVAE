import torch
from scipy.stats import multivariate_normal

from matrix_ops import pack_dense, unpack_dense, is_posdef
from .distribution import ExpDistribution
from torch.distributions import MultivariateNormal


def sample(loc, Sigma, n=1):
    """

    Parameters
    ----------
    loc:
        location parameter
    Sigma:
        scale parameter
    n:
        number of parameters

    If passing an array of loc and Sigma this function will be repeated for every element.

    Returns
    -------

    """
    if len(loc.shape) == 2:
        return [multivariate_normal.rvs(l, S, size=n) for (l, S) in zip(loc, Sigma)]
    return multivariate_normal.rvs(loc, Sigma, size=n)


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


def standard_to_natural(loc, scale):
    scale_inv = torch.inverse(scale)
    eta_1 = torch.bmm(scale_inv, loc[..., None]).squeeze(-1)
    eta_2 = -1 / 2 * scale_inv


def standard_to_info(loc, scale):
    J = torch.inverse(scale)
    # h = torch.solve(scale, loc)
    h = J @ loc
    return J, h


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

        if not torch.isclose(eta_2.squeeze(), eta_2.squeeze().mT, atol=1e-6).all(-2).all(-1):
            print(eta_2.squeeze().cpu().detach().numpy())
            raise ValueError("(natural) Scale matrix not symmetric")

        L = torch.linalg.cholesky(-2 * eta_2 + 1e-6 * torch.eye(len(eta_2), device=eta_2.device))
        # scale = -1 / 2 * torch.inverse(eta_2)
        scale = torch.cholesky_inverse(L)
        loc = torch.bmm(scale, eta_1[..., None]).squeeze()

        return loc, scale

    def rsample(self, n_samples=1):
        """get samples using the re-parameterization trick and natural parameters"""
        loc, scale = self.natural_to_standard()

        if not torch.isclose(scale.squeeze(), scale.squeeze().mT, atol=1e-6).all(-2).all(-1):
            print(scale.squeeze().cpu().detach().numpy())
            raise ValueError("Scale matrix not symmetric")

        if not torch.linalg.cholesky_ex(scale.squeeze()).info.eq(0):
            print(scale.squeeze().cpu().detach().numpy())
            print(sorted(torch.abs(torch.linalg.eigvalsh(scale.squeeze())).cpu().detach().numpy(), reverse=True))
            raise ValueError("Scale matrix not pos eigs")


        return MultivariateNormal(loc, scale.squeeze()).rsample([n_samples])
