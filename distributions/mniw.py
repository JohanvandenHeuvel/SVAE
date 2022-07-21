import torch

from distributions.distribution import ExpDistribution

from .niw import multidigamma


def symmetrize(A):
    return (A + A.T) / 2.0


def is_posdef(A):
    return torch.allclose(A, A.T) and torch.all(torch.linalg.eigvalsh(A) > 0.0)


class MatrixNormalInverseWishart(ExpDistribution):
    def __init__(self, nat_param: torch.Tensor):
        super().__init__(nat_param)
        self.device = nat_param[0].device

    def expected_stats(self):
        K, M, Phi, nu = self.natural_to_standard()

        fudge = 1e-8

        n, _ = M.shape
        p, _ = Phi.shape
        E_T2 = -nu / 2 * symmetrize(torch.inverse(Phi)) + fudge * torch.eye(
            p, device=self.device
        )
        # E_T3 = -2 * torch.matmul(E_T2, M)
        # E_T4 = -0.5 * (symmetrize(torch.matmul(M, E_T3)) + n * K + fudge * torch.eye(p, device=self.device))
        E_T1 = -0.5 * (
            multidigamma(nu / 2.0, n)
            + n * torch.log(torch.tensor([2], device=self.device))
            - torch.slogdet(Phi)[1]
        )

        E_T3 = nu * torch.linalg.solve(Phi, M)
        E_T4 = -0.5 * (
            n * K
            + nu * symmetrize(M.T @ torch.linalg.solve(Phi, M))
            + fudge * torch.eye(n, device=self.device)
        )

        assert is_posdef(-2 * E_T2)
        assert is_posdef(-2 * E_T4)

        return E_T2, E_T3, E_T4, E_T1

    def logZ(self):
        K, M, Phi, nu = self.natural_to_standard()
        p, _ = Phi.shape
        value = (
            p * nu / 2 * torch.log(torch.tensor([2], device=self.device))
            + torch.special.multigammaln(nu / 2, p)
            - nu / 2 * torch.linalg.slogdet(Phi)[1]
            + p / 2 * torch.linalg.slogdet(K)[1]
        )
        return torch.sum(value)

    def standard_to_natural(self, nu, Phi, M, K):
        K_inv = torch.linalg.inv(K)
        A = K_inv
        B = torch.matmul(K_inv, M.T)
        C = Phi + torch.matmul(M, B)
        d = nu
        return A, B, C, d

    def natural_to_standard(self):
        A, B, C, d = self.nat_param
        nu = d
        K_inv = A
        K = symmetrize(torch.linalg.inv(K_inv))
        M = torch.matmul(K, B).T
        Phi = C - torch.matmul(M, B)
        return K, M, Phi, nu
