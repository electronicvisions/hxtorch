"""
Test script for measuring membrane boundaries
"""
import unittest
import hxtorch
from hxtorch.spiking.utils.dynamic_range.boundary import get_dynamic_range
from hxtorch.spiking.utils import calib_helper


class TestBoundaries(unittest.TestCase):
    """ Test for measuring membrane boundaries """

    def test_get_dynamic_range(self):
        # This should use default nightly
        base, upper, lower = get_dynamic_range()
        self.assertTrue(abs(base + 45) < 0.2 * 45)
        self.assertTrue(abs(upper - 100) < 0.2 * 100)
        self.assertTrue(abs(lower + 103) < 0.2 * 102)

        # This should create calib (or use cached calib)
        params = {"tau_mem": 10e-6, "tau_syn": 10e-6}
        base, upper, lower = get_dynamic_range(params=params)
        self.assertTrue(abs(base + 45) < 0.2 * 45)
        self.assertTrue(abs(upper - 100) < 0.2 * 100)
        self.assertTrue(abs(lower + 103) < 0.2 * 102)

        # This should use loaded calibration
        hxtorch.init_hardware()
        calib_path=calib_helper.nightly_calix_native_path()
        hxtorch.release_hardware()
        base, upper, lower = get_dynamic_range(calib_path=calib_path)
        self.assertTrue(abs(base + 45) < 0.2 * 45)
        self.assertTrue(abs(upper - 100) < 0.2 * 100)
        self.assertTrue(abs(lower + 103) < 0.2 * 102)


if __name__ == "__main__":
    unittest.main()
