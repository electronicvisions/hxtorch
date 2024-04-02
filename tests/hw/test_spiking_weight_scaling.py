"""
Test HX util measure_mock_scaling
"""
import unittest
from hxtorch.spiking.utils.dynamic_range.weight_scaling import \
    get_weight_scaling
import hxtorch.spiking.functional as F
from hxtorch.spiking.parameter import MixedHXModelParameter


class TestWeightScaling(unittest.TestCase):
    """ Test script for measuring weight scaling """

    def test_get_weight_scaling(self):
        # This should use default nightly calib indendent of given parameters
        params = {
            "threshold": MixedHXModelParameter(1., 125),
            "tau_mem": 10e-6, "tau_syn": 10e-6}
        weight_scale = get_weight_scaling(params, weight_step=10)
        self.assertAlmostEqual(weight_scale, 50., delta=10)


if __name__ == "__main__":
    unittest.main()
