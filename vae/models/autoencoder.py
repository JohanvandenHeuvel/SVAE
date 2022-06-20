import torch.nn as nn
import torch
import numpy as np

from abc import abstractmethod
import os

from tqdm import tqdm
import pathlib


class Autoencoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def encode(self, x):
        """encode observation x"""
        pass

    @abstractmethod
    def decode(self, z):
        """decode latent z"""
        pass

    @abstractmethod
    def loss_function(self, *args):
        """objective function"""
        pass

    @abstractmethod
    def save_and_log(self, obs, epoch, save_path):
        """function to be called every n epochs"""
        pass

    def save_model(self):
        """save model to disk"""
        path = pathlib.Path().resolve()
        torch.save(self.state_dict(), os.path.join(path, f"{self.name}.pt"))
        print(f"saved model to {os.path.join(path, f'{self.name}.pt')}")

    def load_model(self):
        """load model from disk"""
        path = pathlib.Path().resolve()
        self.load_state_dict(torch.load(os.path.join(path, f"{self.name}.pt")))
        print(f"loaded model from {os.path.join(path, f'{self.name}.pt')}")

    def reparameterize(self, mu, log_var):
        """reparameterization trick for Gaussian"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def kld(self, mu_z, log_var_z):
        """Kullback-Leibler divergence for Gaussian"""
        value = torch.mean(
            -0.5 * torch.sum(1 + log_var_z - mu_z ** 2 - log_var_z.exp(), dim=1), dim=0
        )
        return value

    def fit(
        self, obs, epochs, batch_size, kld_weight, save_path=None, force_train=False
    ):
        """Fit auto-encoder model"""
        self.to(self.device)

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
