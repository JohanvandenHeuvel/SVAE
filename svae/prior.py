from distributions import Dirichlet, NormalInverseWishart
import torch

def flatten_tuple(x):
    return torch.cat((torch.ravel(x[0]), torch.ravel(x[1])))


def prior_logZ(gmm_natparam):
    dirichlet_natparam, niw_natparams = gmm_natparam
    return (
        Dirichlet(dirichlet_natparam).logZ()
        + NormalInverseWishart(niw_natparams).logZ()
    )

def prior_expectedstats(gmm_natparam):
    dirichlet_natparam, niw_natparams = gmm_natparam
    dirichlet_expectedstats = Dirichlet(dirichlet_natparam).expected_stats()
    niw_expectedstats = NormalInverseWishart(niw_natparams).expected_stats()
    return dirichlet_expectedstats, niw_expectedstats


def prior_kl(global_natparam, prior_natparam):
    expected_stats = flatten_tuple(prior_expectedstats(global_natparam))
    natparam_difference = flatten_tuple(global_natparam) - flatten_tuple(prior_natparam)
    logZ_difference = prior_logZ(global_natparam) - prior_logZ(prior_natparam)
    return torch.dot(natparam_difference, expected_stats) - logZ_difference
