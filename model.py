from typing import List

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from addmodule import AddModule


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.001)
        nn.init.normal_(m.bias, mean=0.0, std=0.001)


class resVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):

        super(resVAE, self).__init__()

        """
        ENCODER
        """

        # neural net
        encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        mu_enc = nn.Sequential(encoder, nn.Linear(hidden_size, latent_dim))
        log_var_enc = nn.Sequential(encoder, nn.Linear(hidden_size, latent_dim))

        mu_enc.apply(init_weights)
        log_var_enc.apply(init_weights)

        # linear regression
        mu_enc_identity = nn.Linear(input_size, latent_dim)
        log_var_enc_identity = nn.Linear(input_size, latent_dim)

        # "res net"
        self.mu_enc_res = AddModule(mu_enc, mu_enc_identity)
        self.log_var_enc_res = AddModule(log_var_enc, log_var_enc_identity)

        """
        DECODER 
        """

        # neural net
        decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
        )
        mu_dec = nn.Sequential(decoder, nn.Linear(hidden_size, input_size))
        log_var_dec = nn.Sequential(decoder, nn.Linear(hidden_size, input_size))

        mu_dec.apply(init_weights)
        log_var_dec.apply(init_weights)

        # linear regression
        mu_dec_identity = nn.Linear(latent_dim, input_size)
        log_var_dec_identity = nn.Linear(latent_dim, input_size)

        # "res net"
        self.mu_dec_res = AddModule(mu_dec, mu_dec_identity)
        self.log_var_dec_res = AddModule(log_var_dec, log_var_dec_identity)

    def encode(self, x):
        return self.mu_enc_res(x), self.log_var_enc_res(x)

    def decode(self, z):
        return self.mu_dec_res(z), self.log_var_dec_res(z)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def log_likelihood(self, z, x):
        """
        log-likelihood function over x

        Parameters
        ----------
        z:
            latents
        x:
            observations

        Returns
        -------

        """
        mu, log_var = self.decode(z)

        T, p = mu.shape

        # TODO work out if this is correct
        return -(T * p) / 2 * torch.log(2 * np.pi) + (
            -1 / 2 * (np.sum(((x - mu) / torch.exp(log_var)) ** 2) + np.sum(log_var))
        )

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    # def loss_function(self, x, recon, mu, log_var, kld_weight=1.0):
    #
    #     kld_loss = torch.mean(
    #         -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    #     )
    #     recon_loss = F.mse_loss(recon, x)
    #
    #     loss = recon_loss + kld_weight * kld_loss
    #
    #     return loss

    # def train(self, x_train, epochs, batch_size):
    #
    #     train_loader = torch.utils.data.DataLoader(
    #         x_train, batch_size=batch_size, shuffle=True
    #     )
    #
    #     optimizer = torch.optim.Adam(self.parameters())
    #
    #     train_loss = []
    #     for epoch in range(epochs):
    #
    #         total_loss = []
    #         for x in train_loader:
    #             recon, mu, log_var = self.forward(x.float())
    #
    #             loss = self.loss_function(x.float(), recon, mu, log_var)
    #             total_loss.append(loss.item())
    #
    #             optimizer.zero_grad()
    #             # compute loss
    #             loss.backward()
    #             # update parameters
    #             optimizer.step()
    #
    #         train_loss.append(np.mean(total_loss))
    #
    #         if epoch % 1 == 0:
    #             print(f"Epoch:{epoch}/{epochs} [loss: {train_loss[epoch]:.3f}]")
