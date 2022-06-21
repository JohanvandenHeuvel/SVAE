import torch

from dense import pack_dense, unpack_dense
from .distribution import ExpDistribution


def multidigamma(input, p):
    arr = torch.arange(0, p, device=input.device)
    # input[..., None] - arr[None, ...] is like [input - i for i in range(p)] excluding somme list manipulation
    values = torch.digamma(input[..., None] - arr[None, ...] / 2)
    # sum over values of p
    result = torch.sum(values, dim=-1)
    return result


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
        E_T3 = -2 * torch.bmm(E_T2, mu_0.unsqueeze(2)).squeeze()
        E_T4 = (-1 / 2) * (
            torch.bmm(mu_0.unsqueeze(1), E_T3.unsqueeze(2)).squeeze() + p / kappa
        )
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
        Phi = (
            eta_2 - torch.einsum("bi,bj->bij", (eta_3, eta_3)) / eta_4[..., None, None]
        )
        # nu = eta_1 - p - 2
        nu = eta_1

        return kappa, mu_0, Phi, nu

    def standard_to_natural(self, kappa, mu_0, Phi, nu):
        _, p, _ = Phi.shape

        eta_2 = Phi + kappa * torch.einsum("bi,bj->bij", (mu_0, mu_0))
        eta_3 = kappa * mu_0
        eta_4 = kappa
        # eta_1 = nu + p + 2
        eta_1 = nu

        return pack_dense(eta_2, eta_3, eta_4, eta_1)
