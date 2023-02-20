import matplotlib.pyplot as plt
import torch

from distributions.gaussian import natural_to_info
from figures.helper import get_model, encode, initialize_globals, get_latents, decode
from plot.lds_plot import plot
from run_lds import get_data, data_parameters


def make_plot(obs, decoded_means, prefix, zero_prefix, seq_start=50, seq_end=150):
    def save_fig(fig, title):
        # save to file
        fig.savefig(f"{prefix}_{title}.pdf", bbox_inches='tight', format='pdf')
        plt.close(fig)

    fig = plot(
        obs=obs.cpu().detach().numpy(),
        samples=decoded_means.cpu().detach().numpy(),
        prefix=zero_prefix,
    )

    save_fig(fig, "plot")


def main():
    model = get_model()

    data = get_data(data_parameters)

    data = data[:200]
    data = torch.tensor(data).double()

    zero_prefix = 100
    potentials = encode(data, model, zero_prefix=zero_prefix)
    info_potentials = list(zip(*natural_to_info(potentials)))

    init_param, J11, J12, J22, logZ = initialize_globals(model)

    x = get_latents(init_param, J11, J12, J22, logZ, info_potentials)

    reconstruction, _ = decode(x, model)

    reconstruction = torch.swapaxes(reconstruction, axis0=0, axis1=1)

    make_plot(data, reconstruction, prefix="overview", zero_prefix=zero_prefix)

    print("done")


if __name__ == "__main__":
    main()
