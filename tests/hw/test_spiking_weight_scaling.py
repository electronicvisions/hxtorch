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
            "leak": MixedHXModelParameter(0., 80),
            "reset": MixedHXModelParameter(0., 80),
            "tau_mem": 10e-6, "tau_syn": 10e-6}
        weight_scale = get_weight_scaling(params, weight_step=10)
        reference_value = 50.
        self.assertAlmostEqual(weight_scale, reference_value,
                               delta=0.2*reference_value)


if __name__ == "__main__":
    unittest.main()
