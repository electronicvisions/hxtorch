"""
Test HX util measure_mock_scaling
"""
import unittest
import hxtorch
from hxtorch.spiking.utils.dynamic_range.threshold import get_trace_scaling
from hxtorch.spiking.utils import calib_helper
from hxtorch.spiking.parameter import MixedHXModelParameter


class TestWeightScaling(unittest.TestCase):
    """ Test script for measuring weight scaling """

    def test_get_weight_scaling(self):
        params = {
            "threshold": MixedHXModelParameter(1., 125),
            "tau_mem": 10e-6, "tau_syn": 10e-6}

        trace_scaling = get_trace_scaling(params=params)
        self.assertTrue(abs(trace_scaling - 1 / 45) < 0.007)

        # This should use loaded calibration
        hxtorch.init_hardware()
        calib_path=calib_helper.nightly_calix_native_path()
        hxtorch.release_hardware()
        trace_scaling = get_trace_scaling(
            params=params, calib_path=calib_path)
        self.assertTrue(abs(trace_scaling - 1 / 45) < 0.007)


if __name__ == "__main__":
    unittest.main()
 