import torch

class Gaussian(torch.distributions.Normal):

    def __init__(self, loc, scale):
        super().__init__(loc, scale)

    def expected_stats(self, natural_parameters):

        return 0
