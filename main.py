from model import resVAE
from data import make_pinwheel_data

import torch

import numpy.random as npr

import matplotlib.pyplot as plt


def main(model):
    latent_dim = 2

    # generate synthetic data
    data = make_pinwheel_data(0.3, 0.05, 5, 2000, 0.25)

    model = resVAE(input_size=2, hidden_size=10, latent_dim=latent_dim)
    model.train(data, 20, 32)

    samples = npr.RandomState(0).randn(2 * 2000, latent_dim)

    mu, log_var = model.encode(torch.Tensor(data))
    latents = model.reparameterize(mu, log_var)
    latents = latents.detach().cpu().numpy()

    recon_data, _, _ = model.forward(torch.Tensor(data))
    recon_data = recon_data.detach().cpu().numpy()

    new_data = model.decode(torch.Tensor(samples))
    new_data = new_data.detach().cpu().numpy()

    alpha = 0.3
    markersize = 1.0

    plt.scatter(latents[:, 0], latents[:, 1], alpha=alpha, s=markersize)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=alpha, s=markersize)
    plt.show()

    plt.scatter(data[:, 0], data[:, 1], alpha=alpha, s=markersize)
    plt.scatter(recon_data[:, 0], recon_data[:, 1], alpha=alpha, s=markersize)
    # plt.scatter(new_data[:, 0], new_data[:, 1], alpha=alpha, s=markersize)
    plt.show()

def feval(param):

    pass

def Lfeval(params):
    pass

def SVAE(epochs):

    for epochs in range(epochs):



if __name__ == "__main__":
    main("resvae")
