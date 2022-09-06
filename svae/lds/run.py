import os
import sys

import torch

# add project root to PYTHONPATH
sys.path.append(os.path.join(os.getcwd(), "SVAE"))

from data import make_dot_data
from log import make_folder
from svae.lds import SVAE
from vae import VAE
from hyperparams import SEED
import wandb

LATENT_DIM = 10

hyperparameters = {
    "VAE_parameters": {
        "latent_dim": LATENT_DIM,
        "input_size": 12,
        "hidden_size": [50],
        "recon_loss": "likelihood",
        "name": "vae",
    },
    "SVAE_train_parameters": {
        "batch_size": 80,
        "epochs": 1000,
        "kld_weight": 1.0,
        "latent_dim": LATENT_DIM,
    },
    "data_parameters": {
        "image_width": 12,
        "T": 500,
        "num_steps": 5000,
        "render_sigma": 0.20,
        "v": 0.75,
    },
    "seed": SEED,
}


def data_params():
    def rot(theta):
        s = torch.sin(theta)
        c = torch.cos(theta)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

    N = 2  # size parameter
    A = 0.999 * rot(torch.tensor(2 * torch.pi / 30))
    Q = 0.1 * torch.eye(N)
    C = torch.eye(N)
    R = 0.0001 * torch.eye(N)
    return A, Q, C, R


def get_data():
    # A, Q, C, R = data_params()
    # data = make_lds_data(A, Q, C, R, T=900)
    data = make_dot_data(**hyperparameters["data_parameters"])
    return data


def get_network():
    network = VAE(**hyperparameters["VAE_parameters"])
    return network


def main():
    wandb.init(project="SVAE", config=hyperparameters)

    folder_name = make_folder()

    observations = get_data()
    network = get_network()

    model = SVAE(network, save_path=os.path.join(folder_name, "svae"))
    model.fit(observations, **hyperparameters["SVAE_train_parameters"])


if __name__ == "__main__":
    main()
