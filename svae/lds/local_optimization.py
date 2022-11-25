import random

import numpy as np
import torch

from distributions import MatrixNormalInverseWishart, NormalInverseWishart
from distributions.gaussian import natural_to_info
from matrix_ops import pack_dense
from seed import SEED
from svae.lds.kalman import info_kalman_filter, info_kalman_smoothing
from svae.lds.kalman.sample import info_sample_backward

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


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
