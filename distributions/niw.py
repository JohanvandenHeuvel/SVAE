import torch

from matrix_ops import (
    pack_dense,
    unpack_dense,
    batch_outer_product,
    batch_elementwise_multiplication,
)
from .distribution import ExpDistribution
from distributions import MatrixNormalInverseWishart

from scipy.stats import invwishart, multivariate_normal

from .mniw import multidigamma


def is_batch(t: torch.Tensor):
    """Check if tensor t is two-dimensional"""
    return len(t.shape) == 2


def make_batch(t: torch.Tensor):
    """Add second dimension"""
    if len(t.shape) > 1:
        raise ValueError
    else:
        return t.unsqueeze(1)


def sample(kappa, mu_0, Phi, nu, n=1):
    # first sample Sigma from inverse-wishart
    Sigma = invwishart.rvs(df=nu, scale=Phi, size=n)
    # second sample mu from multivariate-normal
    mu = multivariate_normal.rvs(mu_0, 1 / kappa * Sigma, size=n)
    return mu, Sigma


class NormalInverseWishart(ExpDistribution):
    def __init__(self, nat_param):
        super().__init__(nat_param)
        self.device = nat_param.device

    def expected_stats(self):
        """Compute Expected Statistics

        E[T1] = E[-1/2 log(det(Sigma))]
        E[T2] = E[-1/2 inv(Sigma)]
        E[T3] = E[inv(Sigma) mu]
        E[T4] = E[-1/2 muT inv(Sigma) mu]
        """

        # kappa, mu_0, Phi, nu = self.natural_to_standard()
        #
        # fudge = 1e-6
        # _, p, _ = Phi.shape
        #
        # E_T2 = (
        #     -0.5 * nu[..., None, None] * torch.inverse(Phi)
        #     + fudge * torch.eye(p, device=self.device)[None, ...]
        # )
        # E_T3 = -2 * torch.bmm(E_T2, mu_0.unsqueeze(2)).squeeze(-1)
        # E_T4 = -0.5 * (
        #     torch.bmm(mu_0.unsqueeze(1), E_T3.unsqueeze(2)).squeeze() + p / kappa
        # )
        # # TODO are the signs here correct?
        # E_T1 = -0.5 * (
        #     torch.slogdet(Phi)[1]
        #     - p * torch.log(torch.tensor([2], device=self.device))
        #     - multidigamma(nu / 2, p)
        # )
        #
        # # assert torch.all(torch.linalg.eigvalsh((-2 * E_T2).squeeze()) >= 0.0)
        #
        # return pack_dense(E_T2, E_T3, E_T4, E_T1).squeeze()

        A, b, c, d = unpack_dense(self.nat_param)
        # TODO Phi (S) and K are mixed up?
        mniw = MatrixNormalInverseWishart(
            [
                A.squeeze(),
                b.squeeze().unsqueeze(-1),
                torch.tensor([[c]], device=self.device),
                d,
            ]
        )
        stats = mniw.expected_stats()
        return pack_dense(stats[0], stats[1].squeeze(), stats[2].squeeze(), stats[3])

    def logZ(self):
        # kappa, mu_0, Phi, nu = self.natural_to_standard()
        # p = mu_0.shape[-1]
        # value = (
        #     p * nu / 2 * torch.log(torch.tensor([2], device=self.device))
        #     + torch.special.multigammaln(nu / 2, p)
        #     - nu / 2 * torch.slogdet(Phi)[1]
        #     - p / 2 * torch.log(kappa)
        # )
        # return torch.sum(value)
        A, b, c, d = unpack_dense(self.nat_param)
        mniw = MatrixNormalInverseWishart(
            [
                A.squeeze(),
                b.squeeze().unsqueeze(-1),
                torch.tensor([[c]], device=self.device),
                d,
            ]
        )
        return mniw.logZ()

    def natural_to_standard(self):
        eta_2, eta_3, eta_4, eta_1 = unpack_dense(self.nat_param)

        _, p, _ = eta_2.shape

        kappa = eta_4
        mu_0 = eta_3 / eta_4[..., None]
        Phi = eta_2 - batch_outer_product(eta_3, eta_3) / eta_4[..., None, None]
        # nu = eta_1 - p - 2
        nu = eta_1

        # assert torch.allclose(Phi.squeeze(), Phi.squeeze().T, atol=1e-6)
        # assert torch.all(torch.linalg.eigvalsh(Phi.squeeze()) >= 0.0)

        return kappa, mu_0, Phi, nu

    def standard_to_natural(self, kappa, mu_0, Phi, nu):
        _, p, _ = Phi.shape

        """
        natural_to_standard does not keep the second dimension, so need to add it 
        """
        if not is_batch(kappa):
            kappa = make_batch(kappa)

        # TODO why is 1/kappa used here?
        k = 1 / kappa

        eta_2 = Phi + batch_elementwise_multiplication(
            k, batch_outer_product(mu_0, mu_0)
        )
        eta_3 = k * mu_0
        eta_4 = k
        # eta_1 = nu + p + 2
        eta_1 = nu

        return pack_dense(eta_2, eta_3, eta_4, eta_1)

    def sample(self, labels, n=1):
        """get n samples from the k-th cluster"""
        kappa, mu_0, Phi, nu = self.natural_to_standard()

        kappa = kappa.cpu().detach().numpy()
        mu_0 = mu_0.cpu().detach().numpy()
        Phi = Phi.cpu().detach().numpy()
        nu = nu.cpu().detach().numpy()

        return [sample(kappa[k], mu_0[k], Phi[k], nu[k], n) for k in labels]
