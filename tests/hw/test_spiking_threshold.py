"""
Test HX util measure_mock_scaling
"""
import unittest
import hxtorch
from hxtorch.spiking.utils.dynamic_range.threshold import get_trace_scaling
from hxtorch.core.utils import calib_helper
from hxtorch.core.parameter import MixedHXModelParameter


class TestTraceScaling(unittest.TestCase):
    """ Test script for measuring trace scaling """

    def test_get_trace_scaling(self):
        params = {
            "threshold": MixedHXModelParameter(1., 125),
            "tau_mem": 10e-6,
            "tau_syn": 10e-6,
        }

        # This should use loaded calibration
        hxtorch.init_hardware()
        calib_path = calib_helper.nightly_calib_path()
        hxtorch.release_hardware()
        trace_scaling = get_trace_scaling(
            params=params,
            calib_path=calib_path,
        )
        self.assertLess(abs(trace_scaling - 1 / 45), 0.007)


if __name__ == "__main__":
    unittest.main()
