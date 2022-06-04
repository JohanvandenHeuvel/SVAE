from __future__ import division
import autograd.numpy as np
from autograd.scipy.special import multigammaln, digamma
from functools import partial


T = lambda X: np.swapaxes(X, axis1=-1, axis2=-2)
symmetrize = lambda X: (X + T(X)) / 2.0
outer = lambda x, y: x[..., :, None] * y[..., None, :]
vs, hs = partial(np.concatenate, axis=-2), partial(np.concatenate, axis=-1)


class NormalInverseWishart:
    def __init__(self):
        pass

    def expected_stats(self, natparam, fudge=1e-8):
        S, m, kappa, nu = self.natural_to_standard(natparam)
        d = m.shape[-1]

        E_J = nu[..., None, None] * symmetrize(np.linalg.inv(S)) + fudge * np.eye(d)
        E_h = np.matmul(E_J, m[..., None])[..., 0]
        E_hTJinvh = d / kappa + np.matmul(m[..., None, :], E_h[..., None])[..., 0, 0]
        E_logdetJ = (
            np.sum(digamma((nu[..., None] - np.arange(d)[None, ...]) / 2.0), -1)
            + d * np.log(2.0)
        ) - np.linalg.slogdet(S)[1]

        return pack_dense(
            -1.0 / 2 * E_J, E_h, -1.0 / 2 * E_hTJinvh, 1.0 / 2 * E_logdetJ
        )

    def logZ(self, natparam):
        S, m, kappa, nu = self.natural_to_standard(natparam)
        d = m.shape[-1]
        return np.sum(
            d * nu / 2.0 * np.log(2.0)
            + multigammaln(nu / 2.0, d)
            - nu / 2.0 * np.linalg.slogdet(S)[1]
            - d / 2.0 * np.log(kappa)
        )

    def natural_to_standard(self, natparam):
        A, b, kappa, nu = unpack_dense(natparam)
        m = b / np.expand_dims(kappa, -1)
        S = A - outer(b, m)
        return S, m, kappa, nu

    def standard_to_natural(self, S, m, kappa, nu):
        b = np.expand_dims(kappa, -1) * m
        A = S + outer(b, m)
        return pack_dense(A, b, kappa, nu)


def pack_dense(A, b, *args):
    """Used for packing Gaussian natural parameters and statistics into a dense
    ndarray so that we can use tensordot for all the linear contraction ops."""
    # we don't use a symmetric embedding because factors of 1/2 on h are a pain
    leading_dim, N = b.shape[:-1], b.shape[-1]
    z1, z2 = np.zeros(leading_dim + (N, 1)), np.zeros(leading_dim + (1, 1))
    c, d = args if args else (z2, z2)

    A = A[..., None] * np.eye(N)[None, ...] if A.ndim == b.ndim else A
    b = b[..., None]
    c, d = np.reshape(c, leading_dim + (1, 1)), np.reshape(d, leading_dim + (1, 1))

    return vs((hs((A, b, z1)), hs((T(z1), c, z2)), hs((T(z1), z2, d))))


def unpack_dense(arr):
    N = arr.shape[-1] - 2
    return arr[..., :N, :N], arr[..., :N, N], arr[..., N, N], arr[..., N + 1, N + 1]
