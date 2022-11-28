import torch

from distributions import MatrixNormalInverseWishart
from distributions.gaussian import standard_pair_params
from distributions.mniw import standard_to_natural
from matrix_ops import unpack_dense, pack_dense
from svae.lds import SVAE
from vae import VAE

import wandb
from run_lds import get_data

import matplotlib.pyplot as plt

from plot.lds_plot import (
    plot_latents,
    plot_potentials,
    plot_info_parameters,
    plot_global,
    plot_standard_params_Sigma,
    plot_standard_params_mu,
)

from svae.lds.local_optimization import local_optimization

vae_parameters = {
    "latent_dim": 10,
    "input_size": 12,
    "hidden_size": [50],
    "recon_loss": "likelihood",
    "name": "vae",
    "weight_init_std": 1e-2,
}


def plot_obs(obs):
    """Plot the observations."""
    fig, ax = plt.subplots(1, 1, figsize=(280, 20))
    ax.matshow(obs.T, cmap="gray")
    # ax.axis("tight")
    # ax.axis("off")
    return fig


def get_network():
    """Get trained network"""
    network = VAE(**vae_parameters)
    model = SVAE(network, save_path=None)
    model.load_model(path="../trained", epoch="trained")
    return model


def main():
    params = ["M", "K", "Phi", "nu"]
    values = [0.1, 0.5, 1.0, 2, 10]
    for param in params:
        for val in values:
            print(f"Running ({param}, {val})")
            wandb.init(project="SVAE_LDS_params", config={"val": val, "param": param})
            wandb.run.name = f"{param}_{val}"

            model = get_network()

            """
            Get data
            """
            observations = get_data()
            observations = observations[:200]
            data = torch.tensor(observations).to(model.vae.device).double()

            """
            Get global parameters
            """
            niw_param, mniw_param = model.eta_theta

            K, M, Phi, nu = MatrixNormalInverseWishart(mniw_param).natural_to_standard()
            if param == "M":
                M = M * val
            elif param == "K":
                K = K * val
            elif param == "Phi":
                Phi = Phi * val
            elif param == "nu":
                nu = nu * val
            mniw_param = standard_to_natural(nu, Phi, M, K)

            J11, J12, J22, _ = MatrixNormalInverseWishart(mniw_param).expected_stats()
            J11 = -2 * J11
            J12 = -1 * J12
            J22 = -2 * J22
            A, Q = standard_pair_params(J11, J12, J22)

            model.eta_theta = niw_param, mniw_param

            """
            Forward pass, zero out after time step t
            """
            potentials = model.encode(data)
            scale, loc, _, _ = unpack_dense(potentials)
            loc[100:] = 0.0
            scale[100:] = 0.0
            potentials = pack_dense(scale, loc)

            x, _, _ = local_optimization(potentials, eta_theta=model.eta_theta, n_samples=50)

            decoded_means, _ = model.decode(x.reshape(-1, x.shape[-1]))
            decoded_means = decoded_means.reshape(*x.shape[:-1], -1)
            decoded_means = torch.swapaxes(decoded_means, axis0=0, axis1=1)

            x = torch.swapaxes(x, axis0=0, axis1=1)

            seq_length = -1

            """
            Plotting
            """
            fig_obs = plot_obs(data.cpu().detach().numpy()[:seq_length])
            fig_samples = plot_obs(torch.mean(decoded_means, dim=0).cpu().detach().numpy()[:seq_length])
            fig_latents = plot_latents(x.cpu().detach().numpy()[:, :seq_length])
            fig_potentials = plot_potentials(potentials, prefix=seq_length)
            fig_info_params = plot_info_parameters(J11, J12, J22, J12.T, A, Q, "info_params")
            fig_global_params = plot_global(mniw_param)
            fig_standard_params_Sigma = plot_standard_params_Sigma(mniw_param)
            fig_standard_params_mu = plot_standard_params_mu(mniw_param)

            """
            Manipulation
            """
            wandb.log(
                {
                    "obs": fig_obs,
                    "samples": fig_samples,
                    "latents": fig_latents,
                    "potentials": fig_potentials,
                    "info_params": fig_info_params,
                    "global_params": fig_global_params,
                    "Sigma": fig_standard_params_Sigma,
                    "mu": fig_standard_params_mu,
                }
            )

            plt.close("all")
            wandb.finish()


if __name__ == "__main__":
    main()
