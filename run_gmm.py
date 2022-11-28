import os

import wandb

from data import make_pinwheel_data
from log import make_folder
from svae.gmm import SVAE
from vae import resVAE, VAE

# parameters for the encoder and decoder
vae_parameters = {
    "latent_dim": 2,
    "input_size": 2,
    "hidden_size": [40],
    "recon_loss": "likelihood",
    "name": "resvae",
    "weight_init_std": 1e-2,
}

# parameters for the SVAE model
svae_parameters = {
    "K": 15,
    "batch_size": 50,
    "epochs": 250,
    "kld_weight": 0.35,
}

# parameters for generating synthetic data
data_parameters = {
    "radial_std": 0.3,
    "tangential_std": 0.05,
    "num_classes": 5,
    "num_per_class": 100,
    "rate": 0.25,
}

# combined parameters
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
    name = hyperparameters["VAE_parameters"]["name"]
    if name == "vae":
        return VAE(**hyperparameters["VAE_parameters"])
    elif name == "resvae":
        return resVAE(**hyperparameters["VAE_parameters"])
    else:
        raise ValueError(f"Network name {name} not recognized!")


def main():
    # logging
    wandb.init(project="test", config=hyperparameters)
    folder_name = make_folder(wandb.run.name)

    # get data and vae model
    observations = get_data()
    network = get_network()

    # SVAE model
    model = SVAE(network, save_path=os.path.join(folder_name, "gmm"))
    model.fit(observations, **hyperparameters["SVAE_parameters"])


if __name__ == "__main__":
    main()
