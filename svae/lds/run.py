import os

from data import make_lds_data, make_dot_data
from log import make_folder, save_dict
from plot.gmm_plot import plot_loss
from svae.lds import SVAE
from vae import resVAE, VAE

LATENT_DIM = 2

hyperparameters = {
    "VAE_parameters": {
        "latent_dim": LATENT_DIM,
        "input_size": 20,
        "hidden_size": 50,
        "recon_loss": "likelihood",
        "name": "vae",
    },
    "SVAE_train_parameters": {"batch_size": 50, "epochs": 5000, "kld_weight": 0.0, "latent_dim": LATENT_DIM},
    "data_parameters": {"image_width": 20, "T": 500, "num_steps": 5000},
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

    # latents, observations = make_lds_data(100, noise_scale=1)
    # observations = observations.unsqueeze(1)
    observations = get_data()

    network = get_network()

    model = SVAE(network)
    train_loss = model.fit(
        observations,
        save_path=os.path.join(folder_name, "svae"),
        **hyperparameters["SVAE_train_parameters"]
    )
    plot_loss(
        train_loss, title="svae_loss", save_path=os.path.join(folder_name, "svae")
    )


if __name__ == "__main__":
    main()
