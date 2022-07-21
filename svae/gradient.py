from torch import Tensor


def natural_gradient(
    stats, eta_theta, eta_theta_prior, N, num_batches,
):
    """
    Natural gradient for the global variational parameters eta_theta

    Parameters
    ----------
    stats:
       Sufficient statistics.
    eta_theta:
        Posterior natural parameters for global variables.
    eta_theta_prior:
        Prior natural parameters for global variables.
    N:
        Number of data-points.
    num_batches:
        Number of batches in the data.
    """

    def nat_grad(prior, posterior, s):
        return -1.0 / N * (prior - posterior + num_batches * s)

    value = []
    for i in range(len(eta_theta)):
        value.append(nat_grad(eta_theta_prior[i], eta_theta[i], stats[i]))

    return value


def gradient_descent(w, grad_w, step_size):
    if not isinstance(w, Tensor):
        return [gradient_descent(w[i], grad_w[i], step_size) for i in range(len(w))]
    return w - step_size * grad_w
