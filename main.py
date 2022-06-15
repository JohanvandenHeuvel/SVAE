from model import resVAE
from svae import SVAE
from data import make_pinwheel_data
from log import make_folder, save_dict
from plot import plot_scatter, plot_loss
import os


hyperparameters = {
    "resVAE_parameters": {"latent_dim": 2, "input_size": 2, "hidden_size": 50,},
    "resVAE_train_parameters": {"epochs": 100, "batch_size": 32,},
    "pinwheel_data_parameters": {
        "radial_std": 0.3,
        "tangential_std": 0.05,
        "num_classes": 5,
        "num_per_class": 100,
        "rate": 0.25,
    },
    "SVAE_train_parameters": {"K": 15, "batch_size": 500, "epochs": 100,},
}


def get_data():
    # generate synthetic data
    data = make_pinwheel_data(**hyperparameters["pinwheel_data_parameters"])
    return data


def get_network(data, save_path):
    network = resVAE(**hyperparameters["resVAE_parameters"])
    train_loss = network.train(
        data, save_path=save_path, **hyperparameters["resVAE_train_parameters"]
    )
    plot_loss(train_loss, title="vae_loss", save_path=save_path)
    return network


def main():
    folder_name = make_folder()
    save_dict(hyperparameters, save_path=folder_name, name="hyperparameters")

    data = get_data()
    plot_scatter(data, title="observations", save_path=folder_name)
    network = get_network(data, save_path=os.path.join(folder_name, "vae"))

    model = SVAE(network)
    model.fit(data, save_path=os.path.join(folder_name, "svae"), **hyperparameters["SVAE_train_parameters"])


if __name__ == "__main__":
    main()
