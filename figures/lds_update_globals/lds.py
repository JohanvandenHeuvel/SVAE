import matplotlib.pyplot as plt
import torch

from data import WindowData
from distributions.gaussian import natural_to_info
from figures.helper import get_model, initialize_globals, encode, get_latents, decode
from plot.lds_plot import plot_latents, plot_observations
from run_lds import get_data
from svae.gradient import SGDOptim, natural_gradient
from svae.lds.global_optimization import initialize_global_lds_parameters


def plot(x, decoded_means, prefix, zero_prefix, seq_start=50, seq_end=150):
    def save_fig(fig, title):
        # save to file
        fig.savefig(f"{prefix}_{title}.pdf", bbox_inches="tight", format="pdf")
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


def fit(
        model,
        obs,
        epochs,
        batch_size,
        latent_dim,
):
    """
    Custom fit function for the LDS model, to just retrain the global variables.

    Parameters
    ----------
    model: LDS model
    obs: observations
    epochs: number of epochs
    batch_size: batch size
    latent_dim: latent dimension

    Returns
    -------

    """
    print("Training the SVAE ...")

    """
    Data setup 
    """
    if not isinstance(obs, torch.Tensor):
        data = torch.tensor(obs)
    else:
        data = obs.clone().detach()
    data = data.to(model.vae.device).double()
    dataloader = torch.utils.data.DataLoader(
        WindowData(data, batch_size), batch_size=1, shuffle=True
    )
    num_batches = len(dataloader)

    """
    Initialize priors 
    """
    niw_prior, mniw_prior = initialize_global_lds_parameters(latent_dim)
    niw_param, mniw_param = model.eta_theta
    mniw_prior, mniw_param = list(mniw_prior), list(mniw_param)
    model.eta_theta = niw_param, mniw_param

    """
    Optimizer setup 
    """
    mniw_optimizer = [
        SGDOptim(step_size=10),
        SGDOptim(step_size=10),
        SGDOptim(step_size=10),
        SGDOptim(step_size=10),
    ]

    """
    Optimization loop 
    """
    for epoch in range(epochs + 1):
        if epoch % 10 == 0:
            print(f"Epoch {epoch}")
        for i, y in enumerate(dataloader):
            y = y.squeeze(0)
            _, _, (_, E_pair_stats), _ = model.forward(y)

            """
            Update global variational parameter eta_theta using natural gradient
            """
            nat_grad_pair = natural_gradient(
                E_pair_stats, model.eta_theta[1], mniw_prior, len(data), num_batches
            )
            mniw_param = [
                mniw_optimizer[i].update(model.eta_theta[1][i], nat_grad_pair[i])
                for i in range(len(nat_grad_pair))
            ]

            mniw_param = list([p.detach() for p in mniw_param])
            for p in mniw_param:
                p.requires_grad = True
            model.eta_theta = niw_param, mniw_param


def main():
    model = get_model()

    # parameters for generating synthetic data
    data_parameters = {
        "image_width": 12,
        "T": 500,
        "num_steps": 5000,
        "render_sigma": 0.20,
        "v": 2.0,
    }

    data = get_data(data_parameters)

    # fit(model, data, epochs=50, batch_size=80, latent_dim=10)

    data = data[:200]
    data = torch.tensor(data).double()

    zero_prefix = 100
    potentials = encode(data, model, zero_prefix=zero_prefix)
    info_potentials = list(zip(*natural_to_info(potentials)))

    init_param, J11, J12, J22, logZ = initialize_globals(model)

    x = get_latents(init_param, J11, J12, J22, logZ, info_potentials)

    reconstruction, _ = decode(x, model)

    reconstruction = torch.swapaxes(reconstruction, axis0=0, axis1=1)
    x = torch.swapaxes(x, axis0=0, axis1=1)
    plot(x, reconstruction, prefix="notretrained", zero_prefix=zero_prefix)


if __name__ == "__main__":
    main()
