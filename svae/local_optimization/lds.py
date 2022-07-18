from typing import Tuple

import torch

from dense import unpack_dense, pack_dense
from distributions import NormalInverseWishart, MatrixNormalInverseWishart
from distributions.gaussian import (
    Gaussian,
    info_to_natural,
    info_to_standard,
    outer_product,
)


def is_psd(mat):
    return bool((mat == mat.T).all() and (torch.eig(mat)[0][:, 0] >= 0).all())


def filter(init_params, pair_params, potentials):
    """
    Kalman filtering using information form.

    Parameters
    ----------
    init_params:
        Initial state.
    pair_params:
        Transition mechanics.
    potentials:
        Observations.

    Returns
    -------

    """
    # initialization
    state = Gaussian(init_params)  # x_{t}

    # filtering loop
    forward_messages = []
    log_norm = 0
    for loc, scale in zip(*potentials):

        # convert potentials to information form
        J_obs = -2 * scale
        h_obs = loc

        # do forward step
        cond_msg, pred_msg = state.predict(
            J_obs,
            h_obs,
            *pair_params,
            h1=torch.zeros_like(h_obs),
            h2=torch.zeros_like(h_obs)
        )
        forward_messages.append((cond_msg, pred_msg))

        # set next state
        state = Gaussian(info_to_natural(*pred_msg))  # x_{t+1}

    return forward_messages, log_norm


def smooth(forward_messages, pair_params):
    """
    Kalman smoothing using information form.

    Parameters
    ----------
    forward_messages:
        Messages produced by Kalman filtering.
    pair_params:
        Transition mechanics.

    Returns
    -------

    """
    # initialization
    (J_smooth, h_smooth), _ = forward_messages[-1]
    state = Gaussian(info_to_natural(J_smooth, h_smooth))
    loc, scale = info_to_standard(J_smooth, h_smooth)
    E_x = loc
    E_xxT = scale + outer_product(E_x, E_x)
    E_xnxT = 0.0

    # smoothing loop
    expected_stats = [(E_x, E_xxT, E_xnxT)]
    for i, (cond_msg, pred_msg) in enumerate(reversed(forward_messages[:-1])):
        E_xn, _, _ = expected_stats[i]

        # do backward step
        J_smooth, h_smooth, stats = state.rst_backward(
            cond_msg, pred_msg, pair_params, E_xn
        )
        expected_stats.append(stats)

        # set previous state
        state = Gaussian(info_to_natural(J_smooth, h_smooth))

    return expected_stats


def process_expected_stats(expected_stats):
    def make_init_stats(a):
        E_x, E_xxT, _ = a
        return E_xxT, E_x, 1.0, 1.0

    def make_pair_stats(a, b):
        E_x, E_xxT, E_xnxT = a
        E_xn, E_xnxnT, _ = b
        return E_xxT, E_xnxT.T, E_xnxnT, 1.0

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

    return E_init_stats, E_pair_stats, E_node_stats


def sample(forward_messages, pair_params):
    J11, J12, _, _ = pair_params

    # initialization
    (J_cond, h_cond), _ = forward_messages[-1]
    next_sample = Gaussian(info_to_natural(J_cond, h_cond)).rsample()

    # sampling loop
    samples = [next_sample]
    for ((J_cond, h_cond), _) in reversed(forward_messages[:-1]):

        # get the parameters for the Gaussian we want to sample from
        state = Gaussian(info_to_natural(J_cond, h_cond))
        J, h = state.condition(J11, (-J12 @ next_sample.T).T.squeeze())

        # get the sample
        state = Gaussian(info_to_natural(J, h))
        next_sample = state.rsample()
        samples.append(next_sample)

    return torch.stack(samples)


def local_optimization(
    potentials: torch.Tensor, eta_theta: Tuple[torch.Tensor, torch.Tensor],
):
    """

    Parameters
    ----------
    potentials:
        Output of the encoder network.
    eta_theta:
        Natural global parameters for Q(theta).

    Returns
    -------

    """
    device = potentials.device
    scale, loc, _, _ = unpack_dense(potentials)

    """
    priors 
    """
    niw_param, mniw_param = eta_theta
    init_param = NormalInverseWishart(niw_param).expected_stats()
    pair_param = MatrixNormalInverseWishart(mniw_param).expected_stats()
    J11, J12, J22, logZ = pair_param

    eta_x = (init_param, pair_param)

    # convert from natural to information form
    J11 = -2 * J11
    J12 = -1 * J12
    J22 = -2 * J22

    assert is_psd(J11)
    assert is_psd(J22)

    """
    optimize local parameters
    """
    forward_messages, log_norm = filter(
        init_param, pair_params=(J11, J12, J22, logZ), potentials=(loc, scale)
    )
    expected_stats = smooth(forward_messages, pair_params=(J11, J12, J22, logZ))
    expected_stats = process_expected_stats(list(reversed(expected_stats)))
    samples = sample(forward_messages, pair_params=(J11, J12, J22, logZ))

    """
    Statistics
    """
    global_expected_stats = expected_stats[:-1]
    global_expected_stats = (
        pack_dense(
            global_expected_stats[0][0],
            global_expected_stats[0][1],
            torch.tensor([global_expected_stats[0][2]], device=device),
            torch.tensor([global_expected_stats[0][3]], device=device),
        ),
        global_expected_stats[1],
    )

    local_expected_stats = expected_stats[-1]
    local_expected_stats = pack_dense(
        torch.stack(local_expected_stats[0]), torch.stack(local_expected_stats[1]),
    )
    """
    KL-Divergence 
    """
    local_kld = torch.tensordot(potentials, local_expected_stats, 3) - log_norm

    return eta_x, samples, global_expected_stats, local_kld
