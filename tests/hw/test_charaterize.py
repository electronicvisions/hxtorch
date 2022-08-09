"""
Test weight-scaling script
"""
import unittest
from pathlib import Path

from hxtorch.snn import utils
import hxtorch.snn.functional as F


class TestScript(unittest.TestCase):

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)


class TestMeasureBaselines(TestScript):
    """ Test measure baseline """

    def test_measure_baselines(self):
        handler = utils.MeasureBaseline()
        handler.run()
        handler.plot(self.plot_path.joinpath("baselines.png"))


class TestMeasureUpperBoundary(TestScript):
    """ Test measure upper CADC boundaries """

    def test_measure_upper(self):
        handler = utils.MeasureUpperBoundary()
        handler.run()
        handler.plot(self.plot_path.joinpath("upper.png"))


class TestMeasureLowerBoundary(TestScript):
    """ Test measure lower CADC boundaries """

    def test_measure_lower(self):
        handler = utils.MeasureLowerBoundary()
        handler.run()
        handler.plot(self.plot_path.joinpath("lower.png"))


class TestMeasureThresholds(TestScript):
    """ Test measure thresholds """

    def test_measure_thresholds(self):
        handler = utils.MeasureThreshold()
        handler.run()
        handler.plot(self.plot_path.joinpath("thresholds.png"))


class TestMeasureWeightScaling(TestScript):
    """ Test measure weight scaling between SW and HW traces """

    def test_weight_scaling(self):
        params = F.LIParams(1. / 10e-6, 1. / 10e-6, dt=1e-7)
        handler = utils.MeasureWeightScaling()
        handler.run(params, weight_step=20)
        handler.plot(self.plot_path.joinpath("weight_scaling.png"))


class TestMeasureTraceShift(TestScript):
    """ Test measure trace shift between CADC membrane values and spikes """

    def test_trace_shift(self):
        params = F.LIParams(1. / 10e-6, 1. / 10e-6, dt=1e-6)
        handler = utils.MeasureTraceShift()
        handler.run(params)
        handler.plot(self.plot_path.joinpath("trace_shift.png"))


if __name__ == "__main__":
    unittest.main()
