import os

import wandb

from data import make_dot_data
from log import make_folder
from svae.lds import SVAE
from vae import VAE, resVAE
from seed import SEED

experimental_parameters = {
    "weight_init_std": 1e-2,
    "local_kld_weight": 1.0,
    "global_kld_weight": 1.0,
    "name": "vae",
    "latent_dim": 10,
    "update_init_params": False
}

data_parameters = {
    "image_width": 12,
    "T": 500,
    "num_steps": 5000,
    "render_sigma": 0.20,
    "v": 0.75,
}

vae_parameters = {
    "latent_dim": experimental_parameters["latent_dim"],
    "input_size": data_parameters["image_width"],
    "hidden_size": [50],
    "recon_loss": "likelihood",
    "name": experimental_parameters["name"],
    "weight_init_std": experimental_parameters["weight_init_std"]
}

svae_parameters = {
    "batch_size": 80,
    "epochs": 1000,
    "local_kld_weight": experimental_parameters["local_kld_weight"],
    "global_kld_weight": experimental_parameters["global_kld_weight"],
    "latent_dim": vae_parameters["latent_dim"],
    "update_init_params": experimental_parameters["update_init_params"]
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
    name = hyperparameters["VAE_parameters"]["name"]
    if name == "vae":
        return VAE(**hyperparameters["VAE_parameters"])
    elif name == "resvae":
        return resVAE(**hyperparameters["VAE_parameters"])
    else:
        raise ValueError(f"Network name {name} not recognized!")


def main():
    # logging
    wandb.init(project="SVAE_lds", config=hyperparameters)
    folder_name = make_folder(wandb.run.name)

    # get data and vae model
    observations = get_data()
    network = get_network()

    # SVAE model
    model = SVAE(network, save_path=os.path.join(folder_name, "svae"))
    model.fit(observations, **hyperparameters["SVAE_parameters"])


if __name__ == "__main__":
    main()
