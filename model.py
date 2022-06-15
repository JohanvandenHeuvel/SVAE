import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from addmodule import AddModule
from tqdm import tqdm

from plot import plot_scatter

import os


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

    def loss_function(self, x, recon, mu, log_var, kld_weight=1.0):

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )
        recon_loss = F.mse_loss(recon, x)

        loss = recon_loss + kld_weight * kld_loss

        return loss

    def log_likelihood(self, x, mu, log_var):
        """
        log-likelihood function over x

        Parameters
        ----------
        mu:
            mean
        log_var:
            log variance

        Returns
        -------
            negative log-likelihood, i.e. -log p(x|mu, Sigma)
        """
        var = log_var.exp()

        # Entries of var must be non-negative
        if torch.any(var < 0):
            raise ValueError("var has negative entry/entries")

        # Clamp for stability
        var = var.clone()
        with torch.no_grad():
            var.clamp_(min=1e-6)

        # Calculate the loss
        loss = 0.5 * (torch.log(var) + (x - mu) ** 2 / var)
        loss += 0.5 * np.log(2 * np.pi)
        loss = torch.sum(loss, dim=-1)

        return torch.mean(loss)

    def forward(self, x):
        mu_z, log_var_z = self.encode(x)
        z = reparameterize(mu_z, log_var_z)
        mu_x, log_var_x = self.decode(z)
        recon = reparameterize(mu_x, log_var_x)
        return recon, mu_z, log_var_z

    def train(self, x_train, save_path, epochs, batch_size):

        print("Training the VAE ...")
        os.mkdir(save_path)

        train_loader = torch.utils.data.DataLoader(
            x_train, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.parameters())

        train_loss = []
        for epoch in tqdm(range(epochs)):

            total_loss = []
            for x in train_loader:

                recon, mu, log_var = self.forward(x.float())

                loss = self.loss_function(x.float(), recon, mu, log_var)
                total_loss.append(loss.item())

                optimizer.zero_grad()
                # compute loss
                loss.backward()
                # update parameters
                optimizer.step()

            train_loss.append(np.mean(total_loss))

            if epoch % 1 == 0:
                path = os.path.join(save_path, f"{epoch}")
                os.mkdir(path)

                mu, log_var = self.encode(torch.Tensor(x_train))
                latents = reparameterize(mu, log_var)
                latents = latents.detach().numpy()

                plot_scatter(
                    latents, title=f"vae_latents", save_path=path
                )

        print("Finished training of the VAE")
        return train_loss
