import random

import numpy as np
import torch

from distributions import MatrixNormalInverseWishart, NormalInverseWishart
from distributions.gaussian import (
    info_to_standard,
    Gaussian,
    info_to_natural,
    natural_to_info,
)
from hyperparams import SEED
from matrix_ops import pack_dense, outer_product, symmetrize

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def info_condition(J, h, J_obs, h_obs):
    # if not torch.all(torch.linalg.eigvalsh(J) >= 0.0):
    #     raise ValueError(f"J not pd: {torch.linalg.eigvalsh(J)}")
    return J + J_obs, h + h_obs


def condition(J, h, y, Jxx, Jxy):
    J_cond = J + Jxx
    h_cond = h - (Jxy @ y.T).T
    return J_cond, h_cond


def lognorm(J, h, full=False):
    n = len(h)
    constant = n * torch.log(torch.tensor(2 * torch.pi)) if full else 0.0
    return 0.5 * (h @ torch.linalg.solve(J, h) - torch.slogdet(J)[1] + constant)


def info_marginalize(A, B, C, h1, h2):
    # J = C - B @ inv(C) @ B.T
    J = symmetrize(C - B @ torch.linalg.solve(A, B.T))
    # h = h2 - B @ inv(C) @ h1
    h = h2 - B @ torch.linalg.solve(symmetrize(A), h1)
    return J, h.squeeze()


def info_predict(J, h, J11, J12, J22, logZ):
    J_pred, h_pred = info_marginalize(A=(J + J11), B=J12.T, C=J22, h1=h, h2=0)
    logZ_pred = 0.5 * (
        h @ torch.linalg.solve((J + J11), h) - torch.linalg.slogdet((J + J11))[1]
    )
    return J_pred, h_pred, logZ_pred.squeeze() + logZ


def info_kalman_filter(init_params, pair_params, observations):
    (J, h), logZ = init_params
    J11, J12, J22, logZ_param = pair_params

    if not torch.all(torch.linalg.eigvalsh(J) >= 0.0):
        raise ValueError(f"init J not pd: {torch.linalg.eigvalsh(J)}")

    total_logZ = logZ
    forward_messages = []
    for i, (J_obs, h_obs) in enumerate(observations):
        J_cond, h_cond = info_condition(J, h, J_obs, h_obs)
        J, h, logZ = info_predict(J_cond, h_cond, J11, J12, J22, logZ_param)
        total_logZ += logZ.squeeze()
        forward_messages.append(((J_cond, h_cond), (J, h)))
    total_logZ += lognorm(J, h)
    return forward_messages, total_logZ


def info_rst_smoothing(J, h, cond_msg, pred_msg, pair_params, loc_next):
    J_cond, h_cond = cond_msg
    J_pred, h_pred = pred_msg
    J11, J12, J22 = pair_params

    J_smooth, h_smooth = info_marginalize(
        A=(J - J_pred + J22), B=J12, C=(J_cond + J11), h1=h - h_pred, h2=h_cond
    )

    loc, scale = info_to_standard(J_smooth, h_smooth)

    E_xnxT = -torch.linalg.solve(J - J_pred + J22, J12.T) @ scale + outer_product(
        loc_next, loc
    )
    E_xxT = scale + outer_product(loc, loc)

    stats = (loc, E_xxT, E_xnxT)
    return J_smooth, h_smooth, stats


def process_expected_stats(expected_stats):
    def make_init_stats(a):
        E_x, E_xxT, _ = a
        return (
            E_xxT,
            E_x,
            torch.tensor(1.0, device=E_x.device),
            torch.tensor(1.0, device=E_x.device),
        )

    def make_pair_stats(a, b):
        E_x, E_xxT, E_xnxT = a
        E_xn, E_xnxnT, _ = b
        return E_xxT, E_xnxT.T, E_xnxnT, torch.tensor([1.0], device=E_x.device)

    def make_node_stats(a):
        E_x, E_xxT, _ = a
        return torch.diag(E_xxT), E_x, 1.0

    E_init_stats = make_init_stats(expected_stats[0])
    E_pair_stats = [
        make_pair_stats(a, b) for a, b in zip(expected_stats[:-1], expected_stats[1:])
    ]
    # same pair for every time step
    E_pair_stats = [sum(stats) for stats in list(zip(*E_pair_stats))]

    E_node_stats = [make_node_stats(a) for a in expected_stats]
    E_node_stats = list(zip(*E_node_stats))
    E_node_stats[:2] = [torch.stack(E_stats) for E_stats in E_node_stats[:2]]
    E_node_stats[-1] = torch.tensor(E_node_stats[-1], device=E_node_stats[0].device)

    return E_init_stats, E_pair_stats, E_node_stats


def info_kalman_smoothing(forward_messages, pair_params):
    (J_smooth, h_smooth), _ = forward_messages[-1]
    loc, scale = info_to_standard(J_smooth, h_smooth)
    E_xxT = scale + outer_product(loc, loc)
    E_xnxT = 0.0

    expected_stats = [(loc, E_xxT, E_xnxT)]
    backward_messages = [(J_smooth, h_smooth)]
    for i, (cond_msg, pred_msg) in enumerate(reversed(forward_messages[:-1])):
        loc_next, _, _ = expected_stats[i]
        J_smooth, h_smooth, stats = info_rst_smoothing(
            J_smooth, h_smooth, cond_msg, pred_msg, pair_params, loc_next
        )
        backward_messages.append((J_smooth, h_smooth))
        expected_stats.append(stats)

    expected_stats = process_expected_stats(list(reversed(expected_stats)))
    return list(reversed(backward_messages)), expected_stats


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


def info_observation_params(obs, C, R):
    R_inv = torch.inverse(R)
    R_inv_C = R_inv @ C

    # J_obs = C.T @ inv(R) @ C
    J_obs = C.T @ R_inv_C
    # h_obs = (y - D @ u) @ inv(R) @ C
    h_obs = obs @ R_inv_C

    J_obs = J_obs.unsqueeze(0).repeat(len(obs), 1, 1)
    return zip(J_obs, h_obs)


def info_pair_params(A, Q):
    J22 = torch.inverse(Q)
    J12 = -A.T @ J22
    J11 = A.T @ -J12
    return J11, J12, J22


def local_optimization(potentials, eta_theta, n_samples=1):
    y = list(zip(*natural_to_info(potentials)))

    """
    priors 
    """
    niw_param, mniw_param = eta_theta

    J11, J12, J22, logZ = MatrixNormalInverseWishart(mniw_param).expected_stats()
    J11 = -2 * J11
    J12 = -1 * J12
    J22 = -2 * J22

    local_natparam = NormalInverseWishart(niw_param).expected_stats()
    init_param = natural_to_info(local_natparam), torch.sum(local_natparam[2:])

    """
    optimize local parameters
    """
    forward_messages, logZ = info_kalman_filter(
        init_params=init_param, pair_params=(J11, J12, J22, logZ), observations=y
    )
    _, expected_stats = info_kalman_smoothing(
        forward_messages, pair_params=(J11, J12, J22)
    )
    samples = info_sample_backward(
        forward_messages, pair_params=(J11, J12, J22), n_samples=n_samples
    )

    E_init_stats, E_pair_stats, E_node_stats = expected_stats

    E_node_stats = pack_dense(E_node_stats[0], E_node_stats[1])
    local_kld = torch.tensordot(potentials, E_node_stats, dims=3) - logZ

    return samples, (E_init_stats, E_pair_stats), local_kld
