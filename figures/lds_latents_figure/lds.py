import matplotlib.pyplot as plt
import torch

from distributions.gaussian import natural_to_info
from figures.helper import get_model, encode, initialize_globals, get_latents, decode
from plot.lds_plot import plot_latents, plot_observations
from run_lds import data_parameters, get_data


def plot(x, decoded_means, prefix, zero_prefix, seq_start=50, seq_end=150):
    def save_fig(fig, title):
        # save to file
        fig.savefig(f"{prefix}_{title}.pdf", format='pdf', bbox_inches='tight')
        plt.close(fig)

    fig_latents = plot_latents(
        x.cpu().detach().numpy(),
        xrange=(seq_start, seq_end),
        figsize=(5, 2.5),
        prefix=zero_prefix,
    )

    fig_samples = plot_observations(
        decoded_means[0].cpu().detach().numpy(),
        xrange=(seq_start, seq_end),
        figsize=(5, 2.5),
    )

    save_fig(fig_latents, "latents")
    save_fig(fig_samples, "samples")


def merge_dicts(default, new):
    temp = default.copy()
    for k in new.keys():
        temp[k] = new[k]
    return temp


def main():
    default_params = {"zero_prefix": 100, "shift": 1.5}

    figures = {
        # "baseline": {"params": {}, "data": {}},
        # "nostart": {"params": {"zero_prefix": 0}, "data": {}},
        # "freq": {"params": {}, "data": {"v": 2.0}},
        "shift": {"params": {"shift": 100}, "data": {}},
    }

    for key in figures.keys():
        print(f"Running {key}....")

        # merge parameters
        params = merge_dicts(default_params, figures[key]["params"])
        data_params = merge_dicts(data_parameters, figures[key]["data"])

        model = get_model()

        data = get_data(data_params)
        data = data[:200]
        data = torch.tensor(data).double()

        potentials = encode(data, model, zero_prefix=params["zero_prefix"])
        info_potentials = list(zip(*natural_to_info(potentials)))

        init_param, J11, J12, J22, logZ = initialize_globals(model)

        x = get_latents(init_param, J11, J12, J22, logZ, info_potentials)
        x += params["shift"]

        reconstruction, _ = decode(x, model)

        reconstruction = torch.swapaxes(reconstruction, axis0=0, axis1=1)
        x = torch.swapaxes(x, axis0=0, axis1=1)
        plot(x, reconstruction, prefix=key, zero_prefix=params["zero_prefix"])


if __name__ == "__main__":
    main()
