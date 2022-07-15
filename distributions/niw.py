import torch

from dense import pack_dense, unpack_dense
from .distribution import ExpDistribution

from scipy.stats import invwishart, multivariate_normal


def multidigamma(input, p):
    arr = torch.arange(0, p, device=input.device)
    # input[..., None] - arr[None, ...] is like [input - i for i in range(p)] excluding somme list manipulation
    values = torch.digamma(input[..., None] - arr[None, ...] / 2)
    # sum over values of p
    result = torch.sum(values, dim=-1)
    return result


def is_batch(t: torch.Tensor):
    """Check if tensor t is two-dimensional"""
    return len(t.shape) == 2


def make_batch(t: torch.Tensor):
    """Add second dimension"""
    if len(t.shape) > 1:
        raise ValueError
    else:
        return t.unsqueeze(1)


def batch_outer_product(x, y):
    """Computes xyT.

    e.g. if x.shape = (15, 2) and y.shape = (15, 2)
    then we get that first element of result equals [[x_0 * y_0, x_0 * y_1], [x_1 * y_0, x_1 * y_1]]

    """
    return torch.einsum("bi, bj -> bij", (x, y))


def batch_elementwise_multiplication(x, y):
    """Computes x * y where the fist dimension is the batch, x is a scalar.

    e.g. x.shape = (15, 1), y.shape = (15, 2, 2)
    then we get that first element of result equals x[0] * y[0]

    """
    assert x.shape[1] == 1
    return torch.einsum("ba, bij -> bij", (x, y))


def sample(kappa, mu_0, Phi, nu, n=1):
    # first sample Sigma from inverse-wishart
    Sigma = invwishart.rvs(df=nu, scale=Phi, size=n)
    # second sample mu from multivariate-normal
    mu = multivariate_normal.rvs(
        mu_0, 1 / kappa * Sigma, size=n
    )
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
        kappa, mu_0, Phi, nu = self.natural_to_standard()

        _, p, _ = Phi.shape

        T = lambda A: torch.swapaxes(A, axis0=-1, axis1=-2)
        symmetrize = lambda A: (A + T(A)) / 2

        E_T2 = -nu[..., None, None] / 2 * symmetrize(
            torch.inverse(Phi)
        ) + 1e-8 * torch.eye(p, device=self.device)
        E_T3 = -2 * torch.bmm(E_T2, mu_0.unsqueeze(2)).squeeze(-1)
        E_T4 = (-1 / 2) * (
            torch.bmm(mu_0.unsqueeze(1), E_T3.unsqueeze(2)).squeeze() + p / kappa
        )
        # TODO are the signs here correct?
        E_T1 = (-1 / 2) * (
            torch.slogdet(Phi)[1]
            - p * torch.log(torch.tensor([2], device=self.device))
            - multidigamma(nu / 2, p)
        )

        return pack_dense(E_T2, E_T3, E_T4, E_T1)

    # def logZ(self):
    #     kappa, mu_0, Phi, nu = self.natural_to_standard()
    #
    #     _, p, _ = Phi.shape
    #
    #     value = (
    #         -nu / 2 * torch.slogdet(Phi)[1]
    #         - (nu * p / 2) * torch.log(torch.ones_like(nu) * 2)
    #         + torch.special.multigammaln(nu / 2, p)
    #         + p / 2 * torch.log(2 * torch.pi * 1 / kappa)
    #     )
    #     foo = self._logZ()
    #     return torch.sum(value)

    def logZ(self):
        kappa, mu_0, Phi, nu = self.natural_to_standard()
        p = mu_0.shape[-1]
        value = (
            p * nu / 2 * torch.log(torch.tensor([2], device=self.device))
            + torch.special.multigammaln(nu / 2, p)
            - nu / 2 * torch.slogdet(Phi)[1]
            - p / 2 * torch.log(kappa)
        )
        return torch.sum(value)

    def natural_to_standard(self):
        eta_2, eta_3, eta_4, eta_1 = unpack_dense(self.nat_param)

        _, p, _ = eta_2.shape

        kappa = eta_4
        mu_0 = eta_3 / eta_4[..., None]
        Phi = eta_2 - batch_outer_product(eta_3, eta_3) / eta_4[..., None, None]
        # nu = eta_1 - p - 2
        nu = eta_1

        return kappa, mu_0, Phi, nu

    def standard_to_natural(self, kappa, mu_0, Phi, nu):
        _, p, _ = Phi.shape

        """
        natural_to_standard does not keep the second dimension, so need to add it 
        """
        if not is_batch(kappa):
            kappa = make_batch(kappa)

        eta_2 = Phi + batch_elementwise_multiplication(kappa, batch_outer_product(mu_0, mu_0))
        eta_3 = kappa * mu_0
        eta_4 = kappa
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
