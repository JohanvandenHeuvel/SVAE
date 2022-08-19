import os
import sys

# add project root to PYTHONPATH
sys.path.append(os.path.join(os.getcwd(), "SVAE"))

from data import make_dot_data
from log import make_folder, save_dict
from plot.gmm_plot import plot_loss
from svae.lds import SVAE
from vae import VAE

LATENT_DIM = 8

hyperparameters = {
    "VAE_parameters": {
        "latent_dim": LATENT_DIM,
        "input_size": 20,
        "hidden_size": [50],
        "recon_loss": "likelihood",
        "name": "vae",
    },
    "SVAE_train_parameters": {
        "batch_size": 50,
        "epochs": 20,
        "kld_weight": 0.50,
        "latent_dim": LATENT_DIM,
    },
    "data_parameters": {
        "image_width": 20,
        "T": 500,
        "num_steps": 5000,
        "render_sigma": 0.15,
        "v": 0.75,
    },
}


def get_data():
    data = make_dot_data(**hyperparameters["data_parameters"])
    return data


def get_network():
    network = VAE(**hyperparameters["VAE_parameters"])
    return network


def main():
    folder_name = make_folder()
    save_dict(hyperparameters, save_path=folder_name, name="hyperparameters")

    observations = get_data()

    network = get_network()

    model = SVAE(network, save_path=os.path.join(folder_name, "svae"))
    # model.load_model("/var/tmp/vandenheu/SVAE/svae/lds/results/date:08_19-time:14_37_25/svae", 2)
    train_loss = model.fit(observations, **hyperparameters["SVAE_train_parameters"])
    plot_loss(
        train_loss, title="svae_loss", save_path=os.path.join(folder_name, "svae")
    )


if __name__ == "__main__":
    main()
