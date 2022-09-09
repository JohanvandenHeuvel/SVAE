import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from plot.gmm_plot import plot_reconstruction


def init_weights(l):
    """Initialization for MLP layers.

    E.g. can use: apply(init_weights) on a nn.Sequential object.

    Parameters
    ----------
    l: torch.nn.Module
        layer to initialize.

    Returns
    -------

    """
    if isinstance(l, nn.Linear):
        nn.init.normal_(l.weight, mean=0.0, std=1e-2)
        nn.init.normal_(l.bias, mean=0.0, std=1e-2)


def rand_partial_isometry(m, n):
    """Initialization for skip connections. Makes sure that the output has a similar scale as the input.

    Parameters
    ----------
    m: int
        input size
    n: int
        output size

    Returns
    -------

    """
    d = max(m, n)
    value = np.linalg.qr(np.random.randn(d, d))[0][:m, :n]
    return value


class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, name, recon_loss="MSE"):

        super().__init__()
        self.name = name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.recon_loss = recon_loss

        """
        ENCODER
        """
        encoder_layers = [input_size] + hidden_size
        encoder_modules = nn.ModuleList()
        # hidden layers
        for i in range(len(encoder_layers) - 1):
            encoder_modules.append(nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
            encoder_modules.append(nn.Tanh())
        encoder = nn.Sequential(*encoder_modules)
        # output layer
        self.mu_enc = nn.Sequential(encoder, nn.Linear(encoder_layers[-1], latent_dim))
        self.log_var_enc = nn.Sequential(encoder, nn.Linear(encoder_layers[-1], latent_dim))

        self.mu_enc.apply(init_weights)
        self.log_var_enc.apply(init_weights)

        """
        DECODER
        """
        decoder_layers = [latent_dim] + list(reversed(hidden_size))
        decoder_modules = nn.ModuleList()
        # hidden layers
        for i in range(len(decoder_layers) - 1):
            decoder_modules.append(nn.Linear(decoder_layers[i], decoder_layers[i + 1]))
            decoder_modules.append(nn.Tanh())
        decoder = nn.Sequential(*decoder_modules)
        # output layer
        self.mu_dec = nn.Sequential(decoder, nn.Linear(decoder_layers[-1], input_size))
        self.log_var_dec = nn.Sequential(decoder, nn.Linear(decoder_layers[-1], input_size))

        self.mu_dec.apply(init_weights)
        self.log_var_dec.apply(init_weights)

        self.to(self.device)
        self.double()

    def encode(self, x):
        return self.mu_enc(x), self.log_var_enc(x)

    def decode(self, z):
        return self.mu_dec(z), self.log_var_dec(z)

    def forward(self, x):
        mu_z, log_var_z = self.encode(x)
        z = self.reparameterize(mu_z, log_var_z)
        mu_x, log_var_x = self.decode(z)
        return mu_x, log_var_x, mu_z, log_var_z

    def loss_function(self, x, mu_x, log_var_x, full=False, reduction="mean"):
        if self.recon_loss == "MSE":
            recon_loss = F.mse_loss(mu_x, x)
        elif self.recon_loss == "likelihood":
            recon_loss = F.gaussian_nll_loss(mu_x, x, log_var_x.exp(), full=full, reduction=reduction)
        return recon_loss

    def save_and_log(self, obs, epoch, save_path):
        data = torch.tensor(obs).to(self.device).float()
        mu_z, log_var_z = self.encode(data)
        z = self.reparameterize(mu_z, log_var_z)

        mu_x, log_var_x = self.decode(z)
        plot_reconstruction(
            obs,
            mu_x.cpu().detach().numpy(),
            log_var_x.cpu().detach().numpy(),
            z.cpu().detach().numpy(),
            title=f"{epoch}_vae_recon",
            save_path=save_path,
        )

    def save_model(self, path=None, affix=None):
        """save model to disk"""
        if path is None:
            path = pathlib.Path().resolve()
        name = f"{self.name}_{affix}.pt" if affix is not None else f"{self.name}.pt"
        torch.save(self.state_dict(), os.path.join(path, name))
        # print(f"saved model to {os.path.join(path, name)}")

    def load_model(self, path=None, affix=None):
        """load model from disk"""
        if path is None:
            path = pathlib.Path().resolve()
        name = f"{self.name}_{affix}.pt" if affix is not None else f"{self.name}.pt"
        self.load_state_dict(torch.load(os.path.join(path, name)))
        print(f"loaded model from {os.path.join(path, name)}")

    def reparameterize(self, mu, log_var):
        """reparameterization trick for Gaussian"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def kld(self, mu_z, log_var_z):
        """Kullback-Leibler divergence for Gaussian"""
        value = torch.mean(
            -0.5 * torch.sum(1 + log_var_z - mu_z**2 - log_var_z.exp(), dim=1), dim=0
        )
        return value

    def fit(
        self, obs, epochs, batch_size, kld_weight, save_path=None, force_train=False
    ):
        """Fit auto-encoder model"""

        # Load model if it exists on disk
        if (
            os.path.exists(os.path.join(pathlib.Path().resolve(), f"{self.name}.pt"))
            and not force_train
        ):
            self.load_model()
            return 0

        if save_path is not None:
            os.mkdir(save_path)

        # Make data object
        data = torch.tensor(obs).to(self.device)
        train_loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True
        )

        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters())

        # Start outer training loop, each iter one pass over the whole dataset
        train_loss = []
        for epoch in tqdm(range(epochs)):
            # Start inner training loop, each iter one pass over a single batch
            total_loss = []
            for obs_batch in train_loader:
                obs_batch = obs_batch.float()

                # get values from the model
                mu_x, log_var_x, mu_z, log_var_z = self.forward(obs_batch)

                # reconstruction loss
                recon_loss = self.loss_function(obs_batch, mu_x, log_var_x)
                # regularization
                kld_loss = self.kld(mu_z, log_var_z)
                # loss is combination of above two
                loss = recon_loss + kld_weight * kld_loss

                optimizer.zero_grad()
                # compute gradients
                loss.backward()
                # update parameters
                optimizer.step()

                total_loss.append((recon_loss.item(), kld_loss.item()))
            train_loss.append(np.mean(total_loss, axis=0))

            if epoch % (epochs // 10) == 0:
                self.save_and_log(obs, epoch, save_path)

        self.save_model()
        return train_loss
