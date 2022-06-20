import torch.nn as nn
import torch

from .addmodule import AddModule
from .vae import VAE, init_weights, rand_partial_isometry


class resVAE(VAE):
    """
    An extension of the VAE class. The only difference is that the resVAE uses skip connections:
        I.e. y = f(x) + Ax
    where f is an MLP and A is a matrix. The VAE is just y = f(x). Note that Ax just ends up being linear regression.

    """

    def __init__(self, input_size, hidden_size, latent_dim, name, recon_loss="MSE"):

        super().__init__(input_size, hidden_size, latent_dim, name, recon_loss)

        """
        ENCODER
        """
        # neural net
        encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        mu_enc = nn.Sequential(encoder, nn.Linear(hidden_size, latent_dim))
        log_var_enc = nn.Sequential(encoder, nn.Linear(hidden_size, latent_dim))

        mu_enc.apply(init_weights)
        log_var_enc.apply(init_weights)

        # linear regression
        mu_enc_identity = nn.Linear(input_size, latent_dim, bias=False)
        log_var_enc_identity = nn.Linear(input_size, latent_dim, bias=False)

        mu_enc_identity.weight = nn.Parameter(
            torch.tensor(rand_partial_isometry(input_size, latent_dim)).float()
        )
        log_var_enc_identity.weight = nn.Parameter(
            torch.zeros_like(mu_enc_identity.weight)
        )

        # "res net"
        self.mu_enc = AddModule(mu_enc, mu_enc_identity)
        self.log_var_enc = AddModule(log_var_enc, log_var_enc_identity)

        """
        DECODER 
        """
        # neural net
        decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        mu_dec = nn.Sequential(decoder, nn.Linear(hidden_size, input_size))
        log_var_dec = nn.Sequential(decoder, nn.Linear(hidden_size, input_size))

        mu_dec.apply(init_weights)
        log_var_dec.apply(init_weights)

        # linear regression
        mu_dec_identity = nn.Linear(latent_dim, input_size, bias=False)
        log_var_dec_identity = nn.Linear(latent_dim, input_size, bias=False)

        mu_dec_identity.weight = nn.Parameter(
            torch.tensor(rand_partial_isometry(latent_dim, input_size)).float()
        )
        log_var_dec_identity.weight = nn.Parameter(
            torch.zeros_like(mu_dec_identity.weight)
        )

        # "res net"
        self.mu_dec = AddModule(mu_dec, mu_dec_identity)
        self.log_var_dec = AddModule(log_var_dec, log_var_dec_identity)
