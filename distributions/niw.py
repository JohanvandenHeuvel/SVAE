from .distribution import ExpDistribution

import torch
import numpy as np

from dense import pack_dense, unpack_dense

def multidigamma(input, p):
    values = torch.stack([torch.digamma(input - i/2) for i in range(p)])
    return torch.sum(values, dim=0)


class NormalInverseWishart(ExpDistribution):
    def __init__(self, nat_param):
        super().__init__(nat_param)

    def expected_stats(self):
        """Compute Expected Statistics

        E[T1] = E[-1/2 log(det(Sigma))]
        E[T2] = E[-1/2 inv(Sigma)]
        E[T3] = E[inv(Sigma) mu]
        E[T4] = E[-1/2 muT inv(Sigma) mu]
        """
        kappa, mu_0, Phi, nu = self.natural_to_standard()

        _, p, _ = Phi.shape

        E_T1 = (
            1 / 2 * (torch.slogdet(Phi)[1] - torch.log(torch.Tensor([2])) - multidigamma(nu / 2, p))
        )
        E_T2 = nu[..., None, None] / 2 * torch.inverse(Phi)
        E_T3 = -2 * torch.bmm(E_T2, mu_0.unsqueeze(2)).squeeze()
        E_T4 = torch.bmm(mu_0.unsqueeze(1), E_T3.unsqueeze(2)).squeeze() / -2 + p / 2 * kappa

        return pack_dense(E_T2, E_T3, E_T4, E_T1)

    def logZ(self):
        kappa, mu_0, Phi, nu = self.natural_to_standard()

        _, p, _ = Phi.shape

        value = -nu / 2 * torch.slogdet(Phi)[1]
        (
            -(nu * p / 2) * torch.log(torch.ones_like(nu)*2)
            + torch.special.multigammaln(nu / 2, p)
            + p / 2 * torch.log(2 * torch.pi * 1 / kappa)
        )
        return torch.sum(value)

    def natural_to_standard(self):
        eta_2, eta_3, eta_4, eta_1 = unpack_dense(self.nat_param)

        _, p, _ = eta_2.shape

        kappa = eta_4
        mu_0 = eta_3 / eta_4[..., None]
        Phi = eta_2 - torch.einsum('bi,bj->bij', (eta_3, eta_3)) / eta_4[..., None, None]
        nu = eta_1 - p - 2

        return kappa, mu_0, Phi, nu

    def standard_to_natural(self, kappa, mu_0, Phi, nu):
        _, p, _ = Phi.shape

        eta_2 = Phi + kappa * torch.einsum('bi,bj->bij', (mu_0, mu_0))
        eta_3 = kappa * mu_0
        eta_4 = kappa
        eta_1 = nu + p + 2

        return pack_dense(eta_2, eta_3, eta_4, eta_1)
