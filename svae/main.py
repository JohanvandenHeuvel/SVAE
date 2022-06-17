import os

from vae import resVAE, VAE

from log import make_folder, save_dict
from plot.plot import plot_loss
from data import make_pinwheel_data, make_two_cluster_data
from svae import SVAE

hyperparameters = {
    "VAE_parameters": {"latent_dim": 2, "input_size": 2, "hidden_size": 50, "recon_loss": "likelihood"},
    "VAE_train_parameters": {"epochs": 500, "batch_size": 32, "kld_weight":0.1},
    "pinwheel_data_parameters": {
        "radial_std": 0.3,
        "tangential_std": 0.05,
        "num_classes": 5,
        "num_per_class": 100,
        "rate": 0.25,
    },
    "SVAE_train_parameters": {"K": 2, "batch_size": 50, "epochs": 500, "kld_weight": 0.0},
}


def get_data():
    # generate synthetic data
    # data = make_pinwheel_data(**hyperparameters["pinwheel_data_parameters"])
    data = make_two_cluster_data(100)
    return data


def get_network(data, save_path):
    network = VAE(**hyperparameters["VAE_parameters"])
    train_loss = network.fit(
        data, save_path=save_path, **hyperparameters["VAE_train_parameters"]
    )
    plot_loss(train_loss, title="vae_loss", save_path=save_path)
    return network


def main():
    folder_name = make_folder()
    save_dict(hyperparameters, save_path=folder_name, name="hyperparameters")

    data = get_data()
    network = get_network(data, save_path=os.path.join(folder_name, "vae"))

    model = SVAE(network)
    train_loss = model.fit(data, save_path=os.path.join(folder_name, "svae"), **hyperparameters["SVAE_train_parameters"])
    plot_loss(train_loss, title="svae_loss", save_path=os.path.join(folder_name, "svae"))


if __name__ == "__main__":
    main()
