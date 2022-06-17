import torch.nn as nn

from .addmodule import AddModule
from .vae import VAE


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.001)
        nn.init.normal_(m.bias, mean=0.0, std=0.001)


class resVAE(VAE):
    def __init__(self, input_size, hidden_size, latent_dim, recon_loss="MSE"):

        super().__init__(input_size, hidden_size, latent_dim, recon_loss)

        """
        ENCODER
        """
        # neural net
        encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        mu_enc = nn.Sequential(encoder, nn.Linear(hidden_size, latent_dim))
        log_var_enc = nn.Sequential(encoder, nn.Linear(hidden_size, latent_dim))

        # linear regression
        mu_enc_identity = nn.Linear(input_size, latent_dim)
        log_var_enc_identity = nn.Linear(input_size, latent_dim)

        # "res net"
        self.mu_enc = AddModule(mu_enc, mu_enc_identity)
        self.log_var_enc = AddModule(log_var_enc, log_var_enc_identity)

        """
        DECODER 
        """
        # neural net
        decoder = nn.Sequential(nn.Linear(latent_dim, hidden_size), nn.ReLU())
        mu_dec = nn.Sequential(decoder, nn.Linear(hidden_size, input_size))
        log_var_dec = nn.Sequential(decoder, nn.Linear(hidden_size, input_size))

        # linear regression
        mu_dec_identity = nn.Linear(latent_dim, input_size)
        log_var_dec_identity = nn.Linear(latent_dim, input_size)

        # "res net"
        self.mu_dec = AddModule(mu_dec, mu_dec_identity)
        self.log_var_dec = AddModule(log_var_dec, log_var_dec_identity)

        self.mu_enc.apply(init_weights)
        self.log_var_enc.apply(init_weights)

        self.mu_dec.apply(init_weights)
        self.log_var_dec.apply(init_weights)
