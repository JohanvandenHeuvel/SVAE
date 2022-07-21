import os

from data import make_lds_data
from log import make_folder, save_dict
from plot.gmm_plot import plot_loss
from svae.lds import SVAE
from vae import resVAE

hyperparameters = {
    "VAE_parameters": {
        "latent_dim": 1,
        "hidden_size": 40,
        "recon_loss": "likelihood",
        "name": "vae",
    },
    "SVAE_train_parameters": {"batch_size": 100, "epochs": 1000},
}


def get_network():
    network = resVAE(input_size=1, **hyperparameters["VAE_parameters"])
    return network


def main():
    folder_name = make_folder()
    save_dict(hyperparameters, save_path=folder_name, name="hyperparameters")

    latents, observations = make_lds_data(100, noise_scale=1)

    network = get_network()

    model = SVAE(network)
    train_loss = model.fit(
        (latents, observations),
        save_path=os.path.join(folder_name, "svae"),
        **hyperparameters["SVAE_train_parameters"]
    )
    plot_loss(
        train_loss, title="svae_loss", save_path=os.path.join(folder_name, "svae")
    )


if __name__ == "__main__":
    main()
