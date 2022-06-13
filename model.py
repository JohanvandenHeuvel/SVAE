import numpy as np
import torch
import torch.nn as nn

from addmodule import AddModule


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.001)
        nn.init.normal_(m.bias, mean=0.0, std=0.001)

def init_identity(m):
    nn.init.ones_(m.weight)
    nn.init.ones_(m.bias)


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

        mu_enc_identity.apply(init_weights)
        log_var_enc_identity.apply(init_weights)

        # "res net"
        self.mu_enc_res = AddModule(mu_enc, mu_enc_identity)
        self.log_var_enc_res = AddModule(log_var_enc, log_var_enc_identity)

        """
        DECODER 
        """

        # neural net
        decoder = nn.Sequential(nn.Linear(latent_dim, hidden_size), nn.ReLU(),)
        mu_dec = nn.Sequential(decoder, nn.Linear(hidden_size, input_size))
        log_var_dec = nn.Sequential(decoder, nn.Linear(hidden_size, input_size))

        mu_dec.apply(init_weights)
        log_var_dec.apply(init_weights)

        # linear regression
        mu_dec_identity = nn.Linear(latent_dim, input_size)
        log_var_dec_identity = nn.Linear(latent_dim, input_size)

        mu_dec_identity.apply(init_weights)
        log_var_dec_identity.apply(init_weights)

        # "res net"
        self.mu_dec_res = AddModule(mu_dec, mu_dec_identity)
        self.log_var_dec_res = AddModule(log_var_dec, log_var_dec_identity)

    def encode(self, x):
        return self.mu_enc_res(x), self.log_var_enc_res(x)

    def decode(self, z):
        return self.mu_dec_res(z), self.log_var_dec_res(z)

    def log_likelihood(self, z, x, mu, log_var):
        """
        log-likelihood function over x

        Parameters
        ----------
        z:
            latents
        mu:

        log_var:

        Returns
        -------

        """

        value = np.log(2*np.pi) + torch.sum(log_var + (x - mu)**2 / torch.exp(log_var), dim=-1)

        return -1/2 * torch.mean(value)