import torch

from distributions import MatrixNormalInverseWishart, NormalInverseWishart
from distributions.gaussian import natural_to_info
from matrix_ops import pack_dense, unpack_dense
from run_lds import vae_parameters
from svae.lds import SVAE
from svae.lds.kalman import info_kalman_filter, info_sample_backward
from vae import VAE


def get_model():
    # create the (encoder/decoder) network
    network = VAE(**vae_parameters)

    # create the LDS SVAE
    model = SVAE(network, save_path=None)

    # load trained model parameters for global parameters and the network
    model.load_model(path="../../trained_lds", epoch="trained")

    return model


def encode(data, model, zero_prefix=0):
    # get the encoded data
    potentials = model.encode(data)
    scale, loc, _, _ = unpack_dense(potentials)

    # set the potentials to zero after time step
    if zero_prefix == 0:
        loc = torch.zeros_like(loc)
        scale = torch.zeros_like(scale)
    else:
        loc[zero_prefix:] = 0.0
        scale[zero_prefix:] = 0.0

    # transform the potentials back into convenient form
    potentials = pack_dense(scale, loc)
    return potentials


def decode(x, model):
    # get reconstruction
    mu_y, log_var_y = model.decode(x.reshape(-1, x.shape[-1]))
    mu_y = mu_y.reshape(*x.shape[:-1], -1)
    log_var_y = log_var_y.reshape(*x.shape[:-1], -1)
    return mu_y, log_var_y


def initialize_globals(model):
    niw_param, mniw_param = model.eta_theta

    J11, J12, J22, logZ = MatrixNormalInverseWishart(mniw_param).expected_stats()
    J11 = -2 * J11
    J12 = -1 * J12
    J22 = -2 * J22

    local_natparam = NormalInverseWishart(niw_param).expected_stats()
    init_param = natural_to_info(local_natparam), torch.sum(local_natparam[2:])

    return init_param, J11, J12, J22, logZ


def get_latents(init_param, J11, J12, J22, logZ, obs):
    """
    optimize local parameters
    """
    forward_messages, _ = info_kalman_filter(
        init_params=init_param, pair_params=(J11, J12, J22, logZ), observations=obs
    )
    x = info_sample_backward(
        forward_messages, pair_params=(J11, J12, J22), n_samples=50
    )
    return x
