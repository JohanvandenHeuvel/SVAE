from itertools import cycle, islice, chain, combinations
from typing import Tuple

from distributions import NormalInverseWishart
from distributions.gaussian import Gaussian

import torch
import numpy as np

from distributions.mniw import MatrixNormalInverseWishart


def roundrobin(*iterables):
    """
    Recipe credited to George Sakkis,
    see https://code.activestate.com/recipes/528936-roundrobin-generator/?in=user-2591466

    This recipe implements a round-robin generator, a generator that cycles through N iterables until all of them are exhausted:
    >>> list(roundrobin('abc', [], range(4),  (True,False)))
    ['a', 0, True, 'b', 1, False, 'c', 2, 3]
    """
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


class GaussMarkov(Gaussian):
    """
    Gauss-Markov model with latents x and observations y

    initial:
        p(x_0) = N(x_0| m_0, P_0)

    latent:
        p(x_{t+1} | X_{i <= t}) = N(x_{t+1} | A x_t, Q)

    observations:
        p(y_i | X) = N(y_i| H x_i, R)
    """

    def __init__(
        self,
        nat_param: torch.Tensor,
        A: torch.Tensor,
        Q: torch.Tensor,
        H: torch.Tensor,
        R: torch.Tensor,
    ):
        super().__init__(nat_param)

        self.A = A  # latent transition matrix
        self.Q = Q  # latent noise matrix
        self.H = H  # observation matrix
        self.R = R  # observation noise

    def inference(self, y):

        parameters = []

        # time step 0
        loc, scale = self.natural_to_standard()
        parameters.append((loc, scale))

        # filtering, forward steps
        for i in range(0, len(y), 1):
            # get old parameters
            loc, scale = parameters[i]

            # 1. predict step
            loc, scale = self.predict(loc, scale)

            # 2. filtering step
            loc, scale = self.update(loc, scale, y[i])

            # save new parameters
            parameters.append((loc, scale))

        # smoothing, backward steps
        for i in range(len(y), 1, -1):
            # get parameters at step t
            current_mean, current_cov = parameters[i]

            # get parameters at step t - 1
            previous_mean, previous_cov = parameters[i - 1]

            # 3. smoothing step
            loc, scale = self.smooth(
                previous_mean, current_mean, previous_cov, current_cov
            )

            # save new smoothed parameters
            parameters[i] = (loc, scale)

        return parameters

    def smooth(self, previous_mean, current_mean, previous_cov, current_cov):
        """
        Rauch Tung Striebel smoother

        Parameters
        ----------
        previous_mean:
            Predictive location at time step t.
        current_mean:
            Predictive location at time step t+1.
        previous_cov:
            Predictive scale at time step t.
        current_cov:
            Predictive scale at time step t+1.

        Returns
        -------

        """
        gain = previous_cov @ self.A.T @ np.linalg.inv(current_cov)
        smoothed_mean = previous_mean + gain @ (current_mean - previous_mean)
        smoothed_cov = previous_cov + gain @ (current_cov - previous_cov) @ gain.T

        return smoothed_mean, smoothed_cov

    def update(self, loc, scale, y):
        """
        Compute conditional

        Parameters
        ----------
        loc:
            Predictive mean at time step t.
        scale:
            Predictive scale at time step t.
        y:
            Observation at time step t.

        Returns
        -------

        """

        def using_standard_parameters():
            residual = y - (self.H @ loc)

            temp = np.dot(scale, self.H.T)
            gram_matrix = np.dot(self.H, temp) + self.R
            L = np.linalg.cholesky(gram_matrix)

            gain = solve_triangular(L, temp)

            mu_cond = loc + gain @ residual
            scale_cond = scale - gain @ self.H @ scale

            return mu_cond, scale_cond

        return using_standard_parameters()

    def predict(self, loc, scale):
        """
        Compute the predictive parameters for next time step t+1.

        Parameters
        ----------
        loc:
            Location parameter at time step t.
        scale:
            Scale parameter at time step t.

        Returns
        -------
        parameters at time step t+1.
        """

        def using_standard_parameters():
            predictive_loc = self.A @ loc
            predictive_scale = self.A @ scale @ self.A.T + self.Q

            return predictive_loc, predictive_scale

        return using_standard_parameters()


def local_optimization(
    potentials: torch.Tensor,
    eta_theta: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], float]:
    """

    Parameters
    ----------
    potentials:
        Output of the encoder network.
    eta_theta:
        Natural global parameters for Q(theta).
    epochs:
        Number of epochs to train.

    Returns
    -------

    """

    """
    priors 
    """
    niw_param, mniw_param = eta_theta
    eta_x = (NormalInverseWishart.expected_stats(niw_param), MatrixNormalInverseWishart.expected_stats(mniw_param))

    """
    optimize local parameters
    """
    gauss_markov_model = GaussMarkov(*eta_x)
    samples, expected_stats, local_normalizer = gauss_markov_model.inference(potentials)

    """
    Statistics
    """
    # global_expected_stats, local_expected_stats = (
    #     expected_stats[:-1],
    #     expected_stats[-1],
    # )

    """
    KL-Divergence 
    """
    # local_kld = contract(potentials, local_expected_stats) - local_normalizer

    # return samples, global_expected_stats, local_kld
    return None
