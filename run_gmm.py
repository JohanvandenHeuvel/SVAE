import os

import wandb

from data import make_pinwheel_data
from log import make_folder
from svae.gmm import SVAE
from vae import resVAE

vae_parameters = {
    "latent_dim": 2,
    "input_size": 2,
    "hidden_size": [40],
    "recon_loss": "likelihood",
    "name": "vae",
}

svae_parameters = {
    "K": 15,
    "batch_size": 50,
    "epochs": 10,
    "kld_weight": 0.35,
}

data_parameters = {
    "radial_std": 0.3,
    "tangential_std": 0.05,
    "num_classes": 5,
    "num_per_class": 100,
    "rate": 0.25,
}

hyperparameters = {
    "VAE_parameters": vae_parameters,
    "data_parameters": data_parameters,
    "SVAE_parameters": svae_parameters,
}


def get_data():
    # generate synthetic data
    data = make_pinwheel_data(**hyperparameters["data_parameters"])
    return data


def get_network():
    network = resVAE(**hyperparameters["VAE_parameters"])
    return network


def main():
    # logging
    folder_name = make_folder()
    wandb.init(project="SVAE_gmm", config=hyperparameters)

    # get data and vae model
    data = get_data()
    network = get_network()

    # SVAE model
    model = SVAE(network, save_path=os.path.join(folder_name, "gmm"))
    model.fit(data, **hyperparameters["SVAE_parameters"])


if __name__ == "__main__":
    main()
