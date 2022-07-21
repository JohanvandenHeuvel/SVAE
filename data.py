import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
import torch

from distributions.gaussian import standard_to_natural
from distributions import Gaussian


def make_lds_data(n, noise_scale=1e-1):
    A = torch.diag(torch.ones(1))
    Q = torch.diag(torch.ones(1))
    C = torch.diag(torch.ones(1))
    R = torch.diag(torch.ones(1))

    init_params = standard_to_natural(
        loc=torch.ones(1).unsqueeze(0) * 4,
        scale=torch.diag(torch.ones(1) * noise_scale).unsqueeze(0),
    )
    x_1 = Gaussian(nat_param=init_params).rsample()

    x = [x_1]
    y = []
    for i in range(n):
        old_x = x[i]

        new_x = A @ old_x + Q @ torch.randn(1) * noise_scale
        new_y = C @ old_x + R @ torch.randn(1) * noise_scale

        x.append(new_x)
        y.append(new_y)

    return torch.stack(x[:n]).squeeze(), torch.stack(y).squeeze()


def make_dot_data(
    image_width, T, num_steps, x0=0.0, v=0.5, render_sigma=0.2, noise_sigma=0.1
):
    def triangle(t):
        return sawtooth(np.pi * t, width=0.5)

    def dot_trajectory(t):
        return triangle(v * (t + (1 + x0) / 2))

    def render(x):
        return np.exp(-0.5 * ((x - grid) / render_sigma) ** 2)

    grid = np.linspace(-1, 1, image_width, endpoint=True)
    images = np.vstack(
        [render(dot_trajectory(t)) for t in np.linspace(0, T, num_steps)]
    )
    return images + noise_sigma * npr.randn(*images.shape)


def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    """
    source: https://github.com/mattjj/svae/blob/master/experiments/gmm_svae_synth.py

    """
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = npr.randn(num_classes * num_per_class, 2) * np.array(
        [radial_std, tangential_std]
    )
    features[:, 0] += 1.0
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack(
        [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
    )
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return 10 * npr.permutation(np.einsum("ti,tij->tj", features, rotations))


def make_two_cluster_data(num_per_class):
    """Make two gaussian clusters.

    Parameters
    ----------
    num_per_class: int
        number of samples per class

    Returns
    -------

    """
    mu = np.array([[-1, 1], [1, 1]]) * 4
    Sigma = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]

    data = []
    for m, S in zip(mu, Sigma):
        samples = npr.multivariate_normal(m, S, num_per_class)
        data.append(samples)

    return np.concatenate(data)


if __name__ == "__main__":
    # generate synthetic data
    # data = make_pinwheel_data(0.3, 0.05, 5, 100, 0.25)

    # generate two clusters
    data = make_two_cluster_data(100)

    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
