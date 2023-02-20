import matplotlib.pyplot as plt
import torch

from distributions import MatrixNormalInverseWishart
from distributions import NormalInverseWishart
from distributions.gaussian import natural_to_info
from distributions.gaussian import standard_pair_params, info_pair_params
from figures.helper import encode, get_model
from plot.lds_plot import (
    plot_latents,
    plot_observations, )
from run_lds import data_parameters, get_data
from svae.lds.kalman import info_kalman_filter, info_sample_backward


def initialize_globals(model, latent_i, factor):
    """
    priors
    """
    niw_param, mniw_param = model.eta_theta

    J11, J12, J22, logZ = MatrixNormalInverseWishart(mniw_param).expected_stats()
    J11 = -2 * J11
    J12 = -1 * J12
    J22 = -2 * J22

    # # transform information parameters into standard parameters
    A, Q = standard_pair_params(J11, J12, J22)

    # fig, axs = plt.subplots(1, 1)
    # im = axs.matshow(A.cpu().detach().numpy(), cmap="bwr")
    # # fig.colorbar(im, ax=axs)
    # # axs.title.set_text("A")
    # for i in range(10):
    #     for j in range(10):
    #         c = A[j, i]
    #         axs.text(i, j, f"{c:.2f}", va='center', ha='center')
    # # fig.show()
    # fig.savefig(f"transition_matrix.pdf", format="pdf", bbox_inches='tight', dpi=300)

    A[:, latent_i] *= factor
    # foo = torch.zeros_like(A[latent_i])
    # foo[latent_i] = A[latent_i, latent_i]
    # foo[1] = A[latent_i, 1]
    # A[:, latent_i] = foo


    # fig, axs = plt.subplots(1, 1)
    # im = axs.matshow(A.cpu().detach().numpy())
    # fig.colorbar(im, ax=axs)
    # axs.title.set_text("A")
    # for i in range(10):
    #     for j in range(10):
    #         c = A[j, i]
    #         axs.text(i, j, f"{c:.2f}", va='center', ha='center')
    # fig.show()


    # # transform standard parameters into information parameters
    J11, J12, J22 = info_pair_params(A, Q)

    local_natparam = NormalInverseWishart(niw_param).expected_stats()
    init_param = natural_to_info(local_natparam), torch.sum(local_natparam[2:])

    return init_param, J11, J12, J22, logZ


def get_latents(init_param, J11, J12, J22, logZ, obs):
    """
    optimize local parameters
    """
    forward_messages, _ = info_kalman_filter(
        init_params=init_param, pair_params=(J11, J12, J22, logZ), observations=obs
    )
    x = info_sample_backward(
        forward_messages, pair_params=(J11, J12, J22), n_samples=50
    )
    return x


def decode(x, model):
    # get reconstruction
    mu_y, log_var_y = model.decode(x.reshape(-1, x.shape[-1]))
    mu_y = mu_y.reshape(*x.shape[:-1], -1)
    log_var_y = log_var_y.reshape(*x.shape[:-1], -1)
    return mu_y, log_var_y


def plot(x, decoded_means, prefix, zero_prefix, seq_start=0, seq_end=500):
    def save_fig(fig, title):
        # save to file
        fig.savefig(f"{prefix}_{title}.png")
        plt.close(fig)

    fig_latents = plot_latents(
        x.cpu().detach().numpy(),
        xrange=(seq_start, seq_end),
        figsize=(5, 2.5),
        prefix=zero_prefix,
        # title=f"{prefix}_latents"
    )
    # fig_latents.show()

    fig_samples = plot_observations(
        decoded_means[0].cpu().detach().numpy(),
        xrange=(seq_start, seq_end),
        figsize=(5, 2.5),
        # title=f"{prefix}_samples"
    )
    # fig_samples.show()

    save_fig(fig_latents, "latents")
    save_fig(fig_samples, "samples")


def main():
    mult_factors = [0.8, 0.9, 1.0, 1.10, 1.20]
    # mult_factors = [1.0]

    model = get_model()
    data = get_data(data_parameters)
    data = data[:1000]
    data = torch.tensor(data).double()

    zero_prefix = 100

    potentials = encode(data, model, zero_prefix=zero_prefix)
    y = list(zip(*natural_to_info(potentials)))

    factors = []
    for f in mult_factors:
        print(f"Running factor_{f}....")

        # losses = []
        for latent_i in [6]:
            try:
                """
                manipulate transition matrix
                """
                init_param, J11, J12, J22, logZ = initialize_globals(model, latent_i, f)
                x = get_latents(init_param, J11, J12, J22, logZ, y)

                """
                directly manipulate latent
                """
                # x[:, :, latent_i] *= f

                # transpose
                mu_y, log_var_y = decode(x, model)

                # model.vae.recon_loss = "MSE"
                # recon_loss = (
                #     model.vae.loss_function(
                #         data[:, None, :].repeat(1, 50, 1),
                #         mu_y,
                #         log_var_y,
                #         full=True,
                #         reduction="sum",
                #     )
                #     / x.shape[1]
                # ) / len(data)
                #
                # losses.append(recon_loss.item())

                decoded_means = torch.swapaxes(mu_y, axis0=0, axis1=1)
                x = torch.swapaxes(x, axis0=0, axis1=1)
                torch.save(decoded_means, f"decoded_means_{latent_i}_{f}.pt")
                plot(
                    x, decoded_means, prefix=f"factor{f}_latent{latent_i}", zero_prefix=zero_prefix, seq_start=zero_prefix
                )
            except:
                print(f"crash on latent {latent_i}")
                continue

        # factors.append(losses)

    # labels = [f"{i}" for i in range(1, 11)]
    # x = np.arange(len(labels))
    #
    # fig, ax = plt.subplots()
    # width = 0.15
    # legend = [0.0, 0.1, 1, 10, 100]
    # widths = [width * i for i in range(-2, 3)]
    # rects = [
    #     ax.bar(x + widths[i], factors[i], width, label=f"factor {legend[i]}")
    #     for i in range(5)
    # ]
    # ax.set_xticks(x, labels)
    # ax.set_ylabel("MSE")
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig(f"factor_latents.png")

    print("done")


if __name__ == "__main__":
    main()