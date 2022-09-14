import torch

from distributions import Gaussian
from distributions.gaussian import info_to_natural


def condition(J, h, y, Jxx, Jxy):
    J_cond = J + Jxx
    h_cond = h - (Jxy @ y.T).T
    return J_cond, h_cond


def info_sample_backward(forward_messages, pair_params, n_samples):
    J11, J12, _ = pair_params
    (J_cond, h_cond), _ = forward_messages[-1]
    next_sample = Gaussian(info_to_natural(J_cond, h_cond)).rsample(n_samples=n_samples)

    samples = [next_sample]
    for (J_cond, h_cond), _ in reversed(forward_messages[:-1]):
        J, h = condition(J_cond, h_cond, next_sample, J11, J12)
        # Sample from multiple Gaussians using as mean [h_0, ..., h_i, ...]
        # and block matrix with J on the diagonal as variance.
        _J = torch.kron(torch.eye(len(h), device=J.device), J.contiguous())
        _h = h.flatten()
        next_sample = (
            Gaussian(info_to_natural(_J, _h)).rsample().reshape(len(h), len(J))
        )
        samples.append(next_sample)
    samples = torch.stack(list(reversed(samples)))
    return samples
