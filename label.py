import torch


class Label:
    def __init__(self, K, N):
        """

        Parameters
        ----------
        K: int
            Number of clusters.
        N: int
            Number of data-points.
        """

        self.K = K
        self.N = N
        self.pi = torch.zeros((K, N))

    def reset(self):
        self.pi = torch.ones_like(self.pi) / (self.K + 1e-10)

    def set_parameters(self, log_likelihood, e_dir):

        pi = torch.zeros_like(self.pi)
        pi += log_likelihood
        pi += e_dir.T
        pi -= torch.max(pi, dim=1)
        pi = torch.exp(pi)
        pi = torch.div(torch.sum(pi))

        self.pi = pi

    def assign_label(self):
        """
        Assign labels to data-points.

        Returns
        -------

        """
        return torch.max(self.pi, dim=1)
