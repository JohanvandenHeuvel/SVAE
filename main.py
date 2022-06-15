from model import resVAE
from svae import SVAE
from data import make_pinwheel_data

import torch

import numpy.random as npr

import matplotlib.pyplot as plt


def main(model):
    latent_dim = 2

    # generate synthetic data
    data = make_pinwheel_data(0.3, 0.05, 5, 100, 0.25)

    network = resVAE(input_size=2, hidden_size=10, latent_dim=latent_dim)

    network.train(data, 50, 32)

    model = SVAE(network)
    model.fit(data)


if __name__ == "__main__":
    main("resvae")
