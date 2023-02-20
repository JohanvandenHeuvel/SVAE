"""
File for parameter search
"""
import matplotlib.pyplot as plt
import pandas as pd
import torch

import wandb
from distributions import MatrixNormalInverseWishart
from distributions.mniw import standard_to_natural
from figures.helper import get_model
from matrix_ops import unpack_dense, pack_dense
from run_lds import get_data, data_parameters
from svae.lds.local_optimization import local_optimization


def plot_obs(obs):
    """Plot the observations."""
    fig, ax = plt.subplots(1, 1, figsize=(280, 20))
    ax.matshow(obs.T, cmap="gray")
    # ax.axis("tight")
    # ax.axis("off")
    return fig


def main():
    params = ["M", "K", "Phi", "nu"]
    values = [0.1, 0.5, 1.0, 2, 10]
    results = {}
    for param in params:
        losses = []
        for val in values:
            print(f"Running ({param}, {val})....")
            wandb.init(
                project="SVAE_LDS_params",
                mode="disabled",
                config={"val": val, "param": param},
            )
            wandb.run.name = f"{param}_{val}"

            model = get_model()

            """
            Get data
            """
            observations = get_data(data_parameters)
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

            model.eta_theta = niw_param, mniw_param

            """
            Forward pass, zero out after time step t
            """
            potentials = model.encode(data)
            scale, loc, _, _ = unpack_dense(potentials)
            loc[100:] = 0.0
            scale[100:] = 0.0
            potentials = pack_dense(scale, loc)

            x, _, _ = local_optimization(
                potentials, eta_theta=model.eta_theta, n_samples=50
            )

            mu_y, log_var_y = model.decode(x.reshape(-1, x.shape[-1]))

            recon_loss = (
                model.vae.loss_function(
                    data[:, None, :],
                    mu_y.reshape(*x.shape[:-1], -1),
                    log_var_y.reshape(*x.shape[:-1], -1),
                    full=True,
                    reduction="sum",
                )
                / x.shape[1]
            ) / len(data)
            losses.append(recon_loss.item())
            print(recon_loss.item())

        results[param] = losses

    df = pd.DataFrame.from_dict(results)
    df.plot()
    plt.savefig("globals.png")


if __name__ == "__main__":
    main()
