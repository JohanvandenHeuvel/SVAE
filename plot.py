import matplotlib.pyplot as plt
import numpy as np
import torch

from dense import unpack_dense

from distributions import Dirichlet, NormalInverseWishart


def plot_latents(latents, eta_theta, K=15):
    def get_component(niw_param):
        neghalfJ, h, _, _ = unpack_dense(niw_param)
        J = -2 * neghalfJ
        return torch.linalg.solve(J, h), torch.inverse(J)

    def normalize(arr):
        arr = arr.detach().numpy()
        return np.minimum(1.0, arr / np.sum(arr) * K)

    dir_param, niw_param = eta_theta
    weights = normalize(torch.exp(Dirichlet(dir_param).expected_stats()))
    components = map(get_component, NormalInverseWishart(niw_param).expected_stats())

    "plot latent data-points"
    x, y = zip(*latents.detach().numpy())
    plt.scatter(x, y)

    "plot latent clusters"
    for weight, (mu, Sigma) in zip(weights, components):
        x, y = generate_ellipse(mu.detach().numpy(), Sigma.detach().numpy())
        plt.plot(x, y, alpha=weight, linestyle="-", linewidth=3)

    plt.show()


def generate_ellipse(mu, Sigma):
    t = np.hstack([np.arange(0, 2 * np.pi, 0.01), 0])
    circle = np.vstack([np.sin(t), np.cos(t)])
    ellipse = 2.0 * np.dot(np.linalg.cholesky(Sigma), circle)
    return ellipse[0] + mu[0], ellipse[1] + mu[1]
