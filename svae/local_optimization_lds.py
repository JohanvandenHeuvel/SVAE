from typing import Tuple

import torch

def natural_filter_forward(init_params, pair_params, node_params):


   condition = None
   predict = None

    return messages, log_norm

def natural_smoother(messages, init_params, pair_params, node_params):

    E_init_stats = None
    E_pair_stats = None
    E_node_stats = None

    return E_init_stats, E_pair_stats, E_node_stats

def natural_sample_backward(messages, pair_params):

    samples = None

    return samples

def natural_lds_inference(nat_param, node_params):

    init_params, pair_params = nat_param

    # Forward
    forward_messages, log_norm = natural_filter_forward(init_params, pair_params, node_params)

    # Smooth
    expected_stats = natural_smoother(forward_messages, init_params, pair_params, node_params)

    # Backward
    samples = natural_sample_backward(forward_messages, pair_params)

    return samples, expected_stats, log_norm


def local_optimization(
        potentials: torch.Tensor,
        eta_theta: Tuple[torch.Tensor, torch.Tensor],
        epochs: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], float]:
    """

    Parameters
    ----------
    potentials
    eta_theta
    epochs

    Returns
    -------

    """

    """
    priors 
    """
    niw_param, mniw_param = eta_theta
    eta_x = (niw.expected_stats(niw_param), mniw.expected_stats(mniw_param))

    """
    optimize local parameters
    """
    samples, expected_stats, local_normalizor = natural_lds_inference(
        eta_x, potentials
    )

    """
    Statistics
    """
    global_expected_stats, local_expected_stats = expected_stats[:-1], expected_stats[-1]

    """
    KL-Divergence 
    """
    local_kld = contract(potentials, local_expected_stats) - local_normalizor

    return samples, global_expected_stats, local_kld