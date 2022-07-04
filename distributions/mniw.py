from __future__ import division
import numpy as np
import numpy.random as npr
import torch
# from autograd.scipy.special import multigammaln, digamma
# from autograd.scipy.linalg import solve_triangular
# from autograd import grad
# from autograd.util import make_tuple
from scipy.stats import chi2

from distributions.distribution import ExpDistribution


def symmetrize(A):
    return (A + A.T) / 2.0


def is_posdef(A):
    return np.allclose(A, A.T) and np.all(np.linalg.eigvalsh(A) > 0.0)


class MatrixNormalInverseWishart(ExpDistribution):
    def __init__(self, nat_param: torch.Tensor):
        super().__init__(nat_param)
        self.device = nat_param.device

    # def expectedstats_standard(self, nu, Phi, M, K, fudge=1e-8):
    #     m = M.shape[0]
    #     E_Sigma_inv = nu * symmetrize(torch.inverse(Phi)) + fudge * torch.eye(Phi.shape[0])
    #     E_Sigma_inv_A = nu * torch.linalg.solve(Phi, M)
    #     E_AT_Sigma_inv_A = (
    #         m * K
    #         + nu * symmetrize(np.dot(M.T, np.linalg.solve(Phi, M)))
    #         + fudge * np.eye(K.shape[0])
    #     )
    #     E_logdet_Sigma_inv = (
    #         torch.digamma((nu - np.arange(m)) / 2.0).sum()
    #         + m * np.log(2)
    #         - np.linalg.slogdet(Phi)[1]
    #     )
    #
    #     assert is_posdef(E_Sigma_inv)
    #     assert is_posdef(E_AT_Sigma_inv_A)
    #
    #     return make_tuple(
    #         -1.0 / 2 * E_AT_Sigma_inv_A,
    #         E_Sigma_inv_A.T,
    #         -1.0 / 2 * E_Sigma_inv,
    #         1.0 / 2 * E_logdet_Sigma_inv,
    #     )

    def logZ(self):
        nu, Phi, _, K = self.natural_to_standard()
        p = Phi.shape[0]
        value = (
            p * nu / 2 * torch.log(torch.tensor([2], device=self.device))
            + torch.special.multigammaln(nu / 2, p)
            - nu / 2 * torch.linalg.slogdet(Phi)[1]
            + p / 2 * torch.linalg.slogdet(K)[1]
        )
        return value

    def expected_stats(self):
        return self.expectedstats_standard(*self.natural_to_standard(self.natparam))

    def expected_standard_params(self):
        J11, J12, J22, _ = self.expectedstats()
        A = np.linalg.solve(-2.0 * J22, J12.T)
        Sigma = np.linalg.inv(-2 * J22)
        return A, Sigma

    def standard_to_natural(self, nu, S, M, K):
        Kinv = np.linalg.inv(K)
        A = Kinv
        B = np.dot(Kinv, M.T)
        C = S + np.dot(M, B)
        d = nu
        return (A, B, C, d)

    def natural_to_standard(self):
        A, B, C, d = self.natparam
        nu = d
        Kinv = A
        K = symmetrize(np.linalg.inv(Kinv))
        M = np.dot(K, B).T
        S = C - np.dot(M, B)
        return nu, S, M, K

    def natural_sample(self):
        nu, S, M, K = self.natural_to_standard()

        def sample_invwishart(S, nu):
            n = S.shape[0]
            chol = np.linalg.cholesky(S)

            if (nu <= 81 + n) and (nu == np.round(nu)):
                x = npr.randn(nu, n)
            else:
                x = np.diag(np.sqrt(np.atleast_1d(chi2.rvs(nu - np.arange(n)))))
                x[np.triu_indices_from(x, 1)] = npr.randn(n * (n - 1) // 2)
            R = np.linalg.qr(x, "r")
            T = solve_triangular(R.T, chol.T, lower=True).T
            return np.dot(T, T.T)

        def sample_mn(M, U, V):
            G = npr.normal(size=M.shape)
            return M + np.dot(np.dot(np.linalg.cholesky(U), G), np.linalg.cholesky(V).T)

        Sigma = sample_invwishart(S, nu)
        A = sample_mn(M, Sigma, K)

        return A, Sigma
