from itertools import cycle, islice, chain, combinations
from typing import Tuple

from dense import unpack_dense, pack_dense
from distributions import NormalInverseWishart
from distributions.gaussian import Gaussian, info_to_natural

import torch
import numpy as np

from distributions.mniw import MatrixNormalInverseWishart


def is_psd(mat):
    return bool((mat == mat.T).all() and (torch.eig(mat)[0][:, 0] >= 0).all())


def roundrobin(*iterables):
    """
    Recipe credited to George Sakkis,
    see https://code.activestate.com/recipes/528936-roundrobin-generator/?in=user-2591466

    This recipe implements a round-robin generator, a generator that cycles through N iterables until all of them are exhausted:
    >>> list(roundrobin('abc', [], range(4),  (True,False)))
    ['a', 0, True, 'b', 1, False, 'c', 2, 3]
    """
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def local_optimization(
    potentials: torch.Tensor,
    eta_theta: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], float]:
    """

    Parameters
    ----------
    potentials:
        Output of the encoder network.
    eta_theta:
        Natural global parameters for Q(theta).
    epochs:
        Number of epochs to train.

    Returns
    -------

    """

    def filter(init_params, pair_params, potentials):
        state = Gaussian(init_params)  # x_{t}
        forward_messages = []
        log_norm = 0
        for t, (loc, scale) in enumerate(zip(*potentials)):

            # convert potentials to information form
            J_obs = torch.inverse(scale)
            h_obs = J_obs @ loc

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
        backward_messages = []
        state = Gaussian(info_to_natural(*forward_messages[-1][0]))
        for (cond_msg, pred_msg) in reversed(forward_messages[:1]):
            J, h = state.rst_backward(cond_msg, pred_msg, pair_params)
            backward_messages.append((J, h))
            state = Gaussian(info_to_natural(J, h))
        return backward_messages

    def sample(forward_messages, pair_params):
        J11, J12, _, _ = pair_params

        samples = []
        next_sample = Gaussian(info_to_natural(*forward_messages[-1][0])).rsample()
        samples.append(next_sample)
        for ((J_cond, h_cond), _) in reversed(forward_messages[:-1]):

            # get the parameters for the Gaussian we want to sample from
            state = Gaussian(info_to_natural(J_cond, h_cond))
            J, h = state.condition(J11, (-1 * J12 @ next_sample.T).T)

            # get the sample
            state = Gaussian(info_to_natural(J, h))
            next_sample = state.rsample()
            samples.append(next_sample)

        return samples

    scale, loc, _, _ = unpack_dense(potentials)

    """
    priors 
    """
    niw_param, mniw_param = eta_theta
    init_param = NormalInverseWishart(niw_param).expected_stats()
    J11, J12, J22, logZ = MatrixNormalInverseWishart(mniw_param).expected_stats()

    # convert from natural to information form
    J11 *= -2
    J12 *= -1
    J22 *= -2

    assert is_psd(J11)
    assert is_psd(J22)

    """
    optimize local parameters
    """
    forward_messages, log_norm = filter(
        init_param, pair_params=(J11, J12, J22, logZ), potentials=(loc, scale)
    )
    smooth(forward_messages, pair_params=(J11, J12, J22, logZ))
    samples = sample(forward_messages, pair_params=(J11, J12, J22, logZ))

    """
    Statistics
    """
    # global_expected_stats, local_expected_stats = (
    #     expected_stats[:-1],
    #     expected_stats[-1],
    # )

    """
    KL-Divergence 
    """
    # local_kld = contract(potentials, local_expected_stats) - local_normalizer

    # return samples, global_expected_stats, local_kld
    return None
