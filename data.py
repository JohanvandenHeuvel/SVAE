import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt


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
    data = make_cluster_data(100)

    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
