import numpy as np
import torch
import torch.nn as nn

from addmodule import AddModule


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.001)
        nn.init.normal_(m.bias, mean=0.0, std=0.001)


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


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
        value = -(T * p) / 2 * np.log(2 * np.pi) + (
            -1 / 2 * (torch.sum(((x - mu) / torch.exp(log_var)) ** 2) + torch.sum(log_var))
        )

        return value

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
