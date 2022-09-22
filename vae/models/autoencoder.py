import os
import pathlib
from abc import ABC, abstractmethod

import torch.nn as nn

import torch


class Autoencoder(nn.Module, ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def save_model(self, path=None, affix=None, verbose=False):
        """save model to disk"""
        if path is None:
            path = pathlib.Path().resolve()
        name = f"{self.name}_{affix}.pt" if affix is not None else f"{self.name}.pt"
        torch.save(self.state_dict(), os.path.join(path, name))
        if verbose:
            print(f"saved model to {os.path.join(path, name)}")

    def load_model(self, path=None, affix=None, verbose=False):
        """load model from disk"""
        if path is None:
            path = pathlib.Path().resolve()
        name = f"{self.name}_{affix}.pt" if affix is not None else f"{self.name}.pt"
        self.load_state_dict(torch.load(os.path.join(path, name)))
        if verbose:
            print(f"loaded model from {os.path.join(path, name)}")

    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, z):
        pass

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def forward(self, x):
        pass
