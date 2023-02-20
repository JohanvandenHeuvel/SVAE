import matplotlib.pyplot as plt
import numpy as np
import torch

from distributions.gaussian import natural_to_info
from figures.helper import encode, get_model, initialize_globals, get_latents, decode
from run_lds import data_parameters, get_data


def plot(means, stds, factors):

    n = len(means[0])
    fig, ax = plt.subplots()

    # x-axis
    x = np.arange(n)

    # width of the bars
    width = 0.15
    widths = [width * i for i in range(-2, 3)]

    # plot bars
    rects = [
        ax.bar(x + widths[i], means[i], width, xerr=stds[i], label=f"factor {factors[i]}")
        for i in range(5)
    ]

    # Add some text for labels, title and custom x-axis tick labels, etc.
    labels = [f"{i}" for i in range(1, n + 1)]
    ax.set_xticks(x, labels)
    ax.set_ylabel("MSE")
    ax.set_xlabel("Latent variable")
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"factor_latents.pdf", format="pdf")


def get_loss(model, data, y, latent_i, f, n_iter=10):
    """
    Manipulate transition matrix.

    Parameters
    ----------
    model: SVAE-LDS model
    data: observations
    y: potentials
    latent_i: latent variable index
    f: factor to multiply latent variable with
    n_iter: number of iterations to average over

    Returns
    -------

    """
    init_param, J11, J12, J22, logZ = initialize_globals(model)
    losses = []
    for _ in range(n_iter):
        x = get_latents(init_param, J11, J12, J22, logZ, y)

        # directly manipulate latent
        x[:, :, latent_i] *= f

        mu_y, log_var_y = decode(x, model)

        recon_loss = (
            model.vae.loss_function(
                data[:, None, :].repeat(1, 50, 1),
                mu_y,
                log_var_y,
                full=True,
                reduction="sum",
            )
            / x.shape[1]
        ) / len(data)

        losses.append(recon_loss.item())

    losses = np.array(losses)

    return losses.mean(), losses.std()


def main():
    mult_factors = [0.0, 0.1, 1.0, 10, 100]

    model = get_model()
    model.vae.recon_loss = "MSE" # don't use variance

    data = torch.tensor(get_data(data_parameters)[:1000]).double()
    potentials = encode(data, model, zero_prefix=100)
    y = list(zip(*natural_to_info(potentials)))

    means_per_factor = []
    stds_per_factor = []
    for f in mult_factors:
        print(f"Running factor_{f}....")
        means_per_latent = []
        stds_per_latent = []
        for latent_i in range(10):
            latent_mean, latent_std = get_loss(model, data, y, latent_i, f, n_iter=5)
            means_per_latent.append(latent_mean)
            stds_per_latent.append(latent_std)
        means_per_factor.append(means_per_latent)
        stds_per_factor.append(stds_per_latent)

    plot(means_per_factor, stds_per_factor, mult_factors)
    print("done")


if __name__ == "__main__":
    main()
