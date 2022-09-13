import os

import wandb

from data import make_dot_data
from log import make_folder
from svae.lds import SVAE
from vae import VAE
from seed import SEED

data_parameters = {
    "image_width": 12,
    "T": 500,
    "num_steps": 5000,
    "render_sigma": 0.20,
    "v": 0.75,
}

vae_parameters = {
    "latent_dim": 10,
    "input_size": data_parameters["image_width"],
    "hidden_size": [50],
    "recon_loss": "likelihood",
    "name": "vae",
}

svae_parameters = {
    "batch_size": 80,
    "epochs": 1000,
    "kld_weight": 1.0,
    "latent_dim": vae_parameters["latent_dim"],
}

hyperparameters = {
    "data_parameters": data_parameters,
    "VAE_parameters": vae_parameters,
    "SVAE_parameters": svae_parameters,
    "seed": SEED,
}


def get_data():
    data = make_dot_data(**hyperparameters["data_parameters"])
    return data


def get_network():
    network = VAE(**hyperparameters["VAE_parameters"])
    return network


def main():
    # logging
    folder_name = make_folder()
    wandb.init(project="SVAE", config=hyperparameters)

    # get data and vae model
    observations = get_data()
    network = get_network()

    # SVAE model
    model = SVAE(network, save_path=os.path.join(folder_name, "svae"))
    model.fit(observations, **hyperparameters["SVAE_parameters"])


if __name__ == "__main__":
    main()
