import torch

from distributions.distribution import ExpDistribution
from matrix_ops import symmetrize

from scipy.stats import invwishart, matrix_normal


def sample(M, K, Phi, nu):
    # first sample Sigma from inverse-wishart
    Sigma = invwishart.rvs(df=nu.item(), scale=Phi)
    # second sample A from matrix-normal
    A = matrix_normal.rvs(mean=M, rowcov=Sigma, colcov=K)
    return A, Sigma


def standard_to_natural(nu, Phi, M, K):
    K_inv = torch.linalg.inv(K)
    A = K_inv
    B = torch.matmul(K_inv, M.T)
    C = Phi + torch.matmul(M, B)
    d = nu
    return A, B, C, d


class MatrixNormalInverseWishart(ExpDistribution):
    def __init__(self, nat_param: torch.Tensor):
        super().__init__(nat_param)
        self.device = nat_param[0].device

    def expected_stats(self):
        K, M, Phi, nu = self.natural_to_standard()

        fudge = 1e-8

        n, _ = M.shape
        # E_T2 = -nu / 2 * symmetrize(torch.inverse(Phi)) + fudge * torch.eye(
        #     p, device=self.device
        # )
        # E_T3 = -2 * torch.matmul(E_T2, M)
        # E_T4 = -0.5 * (
        #     symmetrize(torch.matmul(M, E_T3))
        #     + n * K
        #     + fudge * torch.eye(p, device=self.device)
        # )
        # E_T1 = -0.5 * (
        #     torch.slogdet(Phi)[1]
        #     - n * torch.log(torch.tensor([2], device=self.device))
        #     - multidigamma(nu / 2, n)
        # )

        E_T2 = nu * symmetrize(torch.linalg.inv(Phi)) + fudge * torch.eye(
            Phi.shape[0], device=self.device
        )
        E_T3 = nu * torch.linalg.solve(Phi, M)
        E_T4 = (
            n * K
            + nu * symmetrize(torch.matmul(M.T, torch.linalg.solve(Phi, M)))
            + fudge * torch.eye(K.shape[0], device=self.device)
        )
        E_T1 = (
            -torch.slogdet(Phi)[1]
            + n * torch.log(torch.tensor([2], device=self.device))
            + multidigamma(nu / 2, n)
        )

        # assert is_posdef(-2 * E_T2)
        # assert is_posdef(-2 * E_T4)

        # return -0.5 * E_T2, E_T3.T, -0.5 * E_T4, 0.5 * E_T1
        return -0.5 * E_T4, E_T3.T, -0.5 * E_T2, 0.5 * E_T1

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

    def natural_to_standard(self):
        A, B, C, d = self.nat_param
        nu = d
        K_inv = A
        K = symmetrize(torch.linalg.inv(K_inv))
        M = torch.matmul(K, B).T
        Phi = C - torch.matmul(M, B)
        return K, M, Phi, nu

    def sample(self):
        K, M, Phi, nu = self.natural_to_standard()
        return sample(M, K, Phi, nu)


def multidigamma(input, p):
    arr = torch.arange(0, p, device=input.device)
    # input[..., None] - arr[None, ...] is like [input - i for i in range(p)] excluding somme list manipulation
    values = torch.digamma(input[..., None] - arr[None, ...] / 2)
    # sum over values of p
    result = torch.sum(values, dim=-1)
    return result
