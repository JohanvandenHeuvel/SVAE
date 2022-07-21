import os

from data import make_dot_data, make_lds_data
from log import make_folder, save_dict
from plot.gmm_plot import plot_loss
from svae import SVAE
from vae import VAE, resVAE

hyperparameters = {
    "VAE_parameters": {
        "latent_dim": 1,
        "hidden_size": 40,
        "recon_loss": "likelihood",
        "name": "vae",
    },
    "VAE_train_parameters": {"epochs": 500, "batch_size": 32, "kld_weight": 0.1},
    "pinwheel_data_parameters": {
        "radial_std": 0.3,
        "tangential_std": 0.05,
        "num_classes": 5,
        "num_per_class": 100,
        "rate": 0.25,
    },
    "SVAE_train_parameters": {"batch_size": 100, "epochs": 1000,},
}


def get_data():
    data = make_dot_data(
        image_width=20,
        T=500,
        num_steps=5000,
        v=0.75,
        render_sigma=0.15,
        noise_sigma=0.1,
    )
    return data


def get_network(data):
    network = resVAE(input_size=1, **hyperparameters["VAE_parameters"])
    return network


def main():
    folder_name = make_folder()
    save_dict(hyperparameters, save_path=folder_name, name="hyperparameters")

    data = make_lds_data(100, noise_scale=1)

    network = get_network(data[1])

    model = SVAE(network)
    train_loss = model.fit(
        data,
        save_path=os.path.join(folder_name, "svae"),
        **hyperparameters["SVAE_train_parameters"]
    )
    plot_loss(
        train_loss, title="svae_loss", save_path=os.path.join(folder_name, "svae")
    )


if __name__ == "__main__":
    main()
