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

    def __init__(self, input_size, hidden_size, latent_dim, name, weight_init_std, recon_loss="MSE"):

        super().__init__(input_size, hidden_size, latent_dim, name, weight_init_std, recon_loss)

        """
        ENCODER
        """
        encoder_layers = [input_size] + hidden_size
        encoder_modules = nn.ModuleList()
        # hidden layers
        for i in range(len(encoder_layers) - 1):
            encoder_modules.append(nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
            encoder_modules.append(nn.ReLU())
        encoder = nn.Sequential(*encoder_modules)
        # output layer
        mu_enc = nn.Sequential(encoder, nn.Linear(encoder_layers[-1], latent_dim))
        log_var_enc = nn.Sequential(encoder, nn.Linear(encoder_layers[-1], latent_dim))

        mu_enc.apply(lambda l: init_weights(l, weight_init_std))
        log_var_enc.apply(lambda l: init_weights(l, weight_init_std))

        # linear regression
        mu_enc_identity = nn.Linear(input_size, latent_dim, bias=False)
        log_var_enc_identity = nn.Linear(input_size, latent_dim, bias=False)

        mu_enc_identity.weight = nn.Parameter(
            torch.tensor(rand_partial_isometry(input_size, latent_dim)).float().T
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
        decoder_layers = [latent_dim] + list(reversed(hidden_size))
        decoder_modules = nn.ModuleList()
        # hidden layers
        for i in range(len(decoder_layers) - 1):
            decoder_modules.append(nn.Linear(decoder_layers[i], decoder_layers[i + 1]))
            decoder_modules.append(nn.ReLU())
        decoder = nn.Sequential(*decoder_modules)
        # output layer
        mu_dec = nn.Sequential(decoder, nn.Linear(decoder_layers[-1], input_size))
        log_var_dec = nn.Sequential(decoder, nn.Linear(decoder_layers[-1], input_size))

        mu_dec.apply(lambda l: init_weights(l, weight_init_std))
        log_var_dec.apply(lambda l: init_weights(l, weight_init_std))

        # linear regression
        mu_dec_identity = nn.Linear(latent_dim, input_size, bias=False)
        log_var_dec_identity = nn.Linear(latent_dim, input_size, bias=False)

        mu_dec_identity.weight = nn.Parameter(
            torch.tensor(rand_partial_isometry(latent_dim, input_size)).float().T
        )
        log_var_dec_identity.weight = nn.Parameter(
            torch.zeros_like(mu_dec_identity.weight)
        )

        # "res net"
        self.mu_dec = AddModule(mu_dec, mu_dec_identity)
        self.log_var_dec = AddModule(log_var_dec, log_var_dec_identity)

        self.to(self.device)
        self.double()
