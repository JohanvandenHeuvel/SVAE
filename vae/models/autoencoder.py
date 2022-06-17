import torch.nn as nn
import torch
import numpy as np

from abc import abstractmethod
import os

from tqdm import tqdm


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

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
        pass

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def fit(self, obs, epochs, batch_size, save_path, save_every_epoch=10):
        os.mkdir(save_path)

        # Make data object
        train_loader = torch.utils.data.DataLoader(
            obs, batch_size=batch_size, shuffle=True
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

                mu_x, log_var_x, mu_z, log_var_z = self.forward(obs_batch)
                loss = self.loss_function(obs_batch, mu_x, mu_z, log_var_z)

                optimizer.zero_grad()
                # compute gradients
                loss.backward()
                # update parameters
                optimizer.step()

                total_loss.append(loss.item())
            train_loss.append(np.mean(total_loss))

            if epoch % save_every_epoch == 0:
                self.save_and_log(obs, epoch, save_path)

        return train_loss
