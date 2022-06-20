import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from .autoencoder import Autoencoder

from plot.plot import plot_reconstruction


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=1e-2)
        nn.init.normal_(m.bias, mean=0.0, std=1e-2)

    # if isinstance(m, nn.Linear):
    #     # Xavier Initialization for weights
    #     size = m.weight.size()
    #     fan_out = size[0]
    #     fan_in = size[1]
    #     std = np.sqrt(2.0 / (fan_in + fan_out))
    #     m.weight.data.normal_(0.0, std)
    #
    #     # Normal Initialization for Biases
    #     m.bias.data.normal_(0.0, 0.001)

def rand_partial_isometry(m, n):
    d = max(m, n)
    value = np.linalg.qr(np.random.randn(d, d))[0][:m, :n]
    return value

class VAE(Autoencoder):
    def __init__(self, input_size, hidden_size, latent_dim, name, recon_loss="MSE"):

        super(VAE, self).__init__(name)
        self.recon_loss = recon_loss

        """
        ENCODER
        """
        encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.mu_enc = nn.Sequential(encoder, nn.Linear(hidden_size, latent_dim))
        self.log_var_enc = nn.Sequential(encoder, nn.Linear(hidden_size, latent_dim))

        """
        DECODER
        """
        decoder = nn.Sequential(nn.Linear(latent_dim, hidden_size), nn.ReLU())
        self.mu_dec = nn.Sequential(decoder, nn.Linear(hidden_size, input_size))
        self.log_var_dec = nn.Sequential(decoder, nn.Linear(hidden_size, input_size))

        # self.mu_enc.apply(init_weights)
        # self.log_var_enc.apply(init_weights)
        #
        # self.mu_dec.apply(init_weights)
        # self.log_var_dec.apply(init_weights)

    def encode(self, x):
        return self.mu_enc(x), self.log_var_enc(x)

    def decode(self, z):
        return self.mu_dec(z), self.log_var_dec(z)

    def forward(self, x):
        mu_z, log_var_z = self.encode(x)
        z = self.reparameterize(mu_z, log_var_z)
        mu_x, log_var_x = self.decode(z)
        return mu_x, log_var_x, mu_z, log_var_z

    def loss_function(self, x, mu_x, log_var_x):
        if self.recon_loss == "MSE":
            recon_loss = F.mse_loss(mu_x, x)
        elif self.recon_loss == "likelihood":
            recon_loss = F.gaussian_nll_loss(mu_x, x, log_var_x.exp())
        return recon_loss

    def save_and_log(self, obs, epoch, save_path):
        mu_z, log_var_z = self.encode(torch.Tensor(obs))
        z = self.reparameterize(mu_z, log_var_z)

        mu_x, _, _, _ = self.forward(torch.Tensor(obs))
        plot_reconstruction(
            obs,
            mu_x.detach().numpy(),
            z.detach().numpy(),
            title=f"{epoch}_vae_recon",
            save_path=save_path,
        )
