import os

import wandb

from data import make_pinwheel_data
from log import make_folder
from models import VAE, resVAE

data_parameters = {
    "radial_std": 0.3,
    "tangential_std": 0.05,
    "num_classes": 5,
    "num_per_class": 100,
    "rate": 0.25,
}

vae_parameters = {
    "latent_dim": 2,
    "input_size": 2,
    "hidden_size": [50],
    "name": "resvae",
    "recon_loss": "likelihood",
    "weight_init_std": 1e-2,
}

vae_train_parameters = {"epochs": 500, "batch_size": 32, "kld_weight": 0.1}

hyperparameters = {
    "VAE_parameters": vae_parameters,
    "VAE_train_parameters": vae_train_parameters,
    "pinwheel_data_parameters": data_parameters,
}


def get_data():
    # generate synthetic data
    data = make_pinwheel_data(**hyperparameters["pinwheel_data_parameters"])
    return data


def get_network(save_path):
    name = hyperparameters["VAE_parameters"]["name"]
    if name == "vae":
        return VAE(**hyperparameters["VAE_parameters"], save_path=save_path)
    elif name == "resvae":
        return resVAE(**hyperparameters["VAE_parameters"], save_path=save_path)
    else:
        raise ValueError(f"Network name {name} not recognized!")


def main():
    # logging
    wandb.init(project="test", config=hyperparameters)
    folder_name = make_folder(wandb.run.name)

    # get data and vae model
    observations = get_data()
    network = get_network(save_path=os.path.join(folder_name, "vae"))
    network.fit(observations, **hyperparameters["VAE_train_parameters"])
    # plot_loss(train_loss, title="vae_loss", save_path=save_path)


if __name__ == "__main__":
    main()
