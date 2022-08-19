import torch

from distributions import NormalInverseWishart, MatrixNormalInverseWishart, exponential_kld


def initialize_global_lds_parameters(n, scale=1.0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nu = torch.tensor([n + 1])
    Phi = 2 * scale * (n + 1) * torch.eye(n)
    mu_0 = torch.zeros(n)
    kappa = torch.tensor([1 / (2 * scale * n)])

    M = torch.eye(n)
    K = 1 / (2 * scale * n) * torch.eye(n)

    init_state_prior = NormalInverseWishart(torch.zeros_like(nu)).standard_to_natural(
        kappa.unsqueeze(0), mu_0.unsqueeze(0), Phi.unsqueeze(0), nu.unsqueeze(0)
    )
    dynamics_prior = MatrixNormalInverseWishart(
        torch.zeros_like(nu)
    ).standard_to_natural(nu, Phi, M, K)

    dynamics_prior = tuple([d.to(device) for d in dynamics_prior])

    return init_state_prior.to(device), dynamics_prior


def prior_kld_lds(eta_theta, eta_theta_prior):
    niw_params, mniw_params = eta_theta
    niw_params_prior, mniw_params_prior = eta_theta_prior

    niw = NormalInverseWishart(niw_params)
    niw_prior = NormalInverseWishart(niw_params_prior)
    niw_kld = exponential_kld(niw_prior, niw)

    mniw = MatrixNormalInverseWishart(mniw_params)
    mniw_prior = MatrixNormalInverseWishart(mniw_params_prior)
    mniw_kld = exponential_kld(mniw_prior, mniw)

    return mniw_kld + niw_kld
