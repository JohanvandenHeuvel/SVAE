from unittest import TestCase

import torch

from distributions import NormalInverseWishart

from svae.global_optimization import initialize_global_parameters


class TestNormalInverseWishart(TestCase):
    def test_natural_to_standard(self):
        _, niw_natparam = initialize_global_parameters(
            K=15, D=2, alpha=1.0, niw_conc=1.0, random_scale=3.0
        )
        niw = NormalInverseWishart(niw_natparam)

        standard_param = niw.natural_to_standard()
        natural_param = niw.standard_to_natural(*standard_param)

        assert torch.equal(niw_natparam, natural_param)
