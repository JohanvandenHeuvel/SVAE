import torch
from torch import Tensor


def natural_gradient(
    stats, eta_theta, eta_theta_prior, N, num_batches,
):
    """
    Natural gradient for the global variational parameters eta_theta

    Parameters
    ----------
    stats:
       Sufficient statistics.
    eta_theta:
        Posterior natural parameters for global variables.
    eta_theta_prior:
        Prior natural parameters for global variables.
    N:
        Number of data-points.
    num_batches:
        Number of batches in the data.
    """

    def nat_grad(prior, posterior, s):
        return -1.0 / N * (prior - posterior + num_batches * s)

    value = []
    for i in range(len(eta_theta)):
        assert eta_theta[i].shape == stats[i].shape
        value.append(nat_grad(eta_theta_prior[i], eta_theta[i], stats[i]))

    return value


class AdamOptim:
    def __init__(self, step_size, beta1=0.9, beta2=0.999, eps=1e-8):
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = 0.0
        self.v = 0.0

    def update(self, t, w, grad_w):
        if not isinstance(w, Tensor):
            return [self.update(t, w[i], grad_w[i]) for i in range(len(w))]
        self.m = (
            1 - self.beta1
        ) * grad_w + self.beta1 * self.m  # first moment estimate
        self.v = (1 - self.beta2) * (
            grad_w ** 2
        ) + self.beta1 * self.v  # second moment estimate
        m_hat = self.m / (1 - self.beta1 ** (t + 1))
        v_hat = self.v / (1 - self.beta2 ** (t + 1))
        return w - self.step_size * m_hat / (torch.sqrt(v_hat) + self.eps)


class SGDOptim:
    def __init__(self, step_size):
        self.step_size = step_size

    def update(self, w, grad_w):
        if not isinstance(w, Tensor):
            return [self.update(w[i], grad_w[i]) for i in range(len(w))]
        return w - self.step_size * grad_w
