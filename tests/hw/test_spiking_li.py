"""
Test snn.HXNeuron
"""
import unittest
from pathlib import Path

import torch

import hxtorch
from hxtorch import spiking as hxsnn
from hxtorch.core.utils import calib_helper


hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)
logger = hxtorch.logger.get("hxtorch.test.hw.test_spiking_modules")


class HWTestCase(unittest.TestCase):
    """ HW setup """

    dt = 1.0e-6
    plot_path = Path(__file__).parent.joinpath("plots")

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()


class TestLI(HWTestCase):
    """ Test hxtorch.spiking.modules.LI """

    def test_output_type(self):
        """
        Test LI returns the expected handle
        """
        experiment = hxsnn.Experiment()
        neuron = hxsnn.LI(44, experiment)
        # Test output handle
        neuron_handle = neuron(
            hxsnn.SynapseHandle(graded_spikes=torch.zeros(10, 44)))
        self.assertTrue(isinstance(neuron_handle, hxsnn.LIObservables))
        self.assertIsNone(neuron_handle.membrane_cadc)
        self.assertIsNone(neuron_handle.membrane_madc)

    def test_print(self):
        """ Test module printing """
        experiment = hxsnn.Experiment()
        module = hxsnn.LI(10, experiment=experiment)
        logger.INFO(module)

    def test_record_cadc(self):
        """
        Test CADC recording.

        TODO:
            - Ensure correct order.
        """
        experiment = hxsnn.Experiment(dt=self.dt)
        experiment.calibration = calib_helper.fixture_calibration_from_file(
            calib_helper.nightly_calib_path()
        )

        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        li = hxsnn.LI(10, experiment=experiment)

        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63

        spikes = torch.zeros(110, 10, 10)
        for idx in range(10):
            spikes[idx * 10 + 5, :, idx] = 1

        i_handle = linear(hxsnn.LIFObservables(spikes=spikes))
        v_handle = li(i_handle)

        self.assertIsNone(v_handle.membrane_cadc)
        self.assertIsNone(v_handle.membrane_madc)

        hxsnn.run(experiment, 110)

        # Assert types and shapes
        self.assertIsInstance(v_handle.membrane_cadc, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(v_handle.membrane_cadc.shape),
                torch.tensor([110, 10, 10])))
        self.assertIsNone(v_handle.membrane_madc)

    def test_record_madc(self):
        """
        Test MADC recording.

        TODO:
            - Ensure correct neuron is recorded.
        """
        experiment = hxsnn.Experiment(dt=self.dt)
        experiment.calibration = calib_helper.fixture_calibration_from_file(
            calib_helper.nightly_calib_path()
        )

        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        li = hxsnn.LI(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)

        spikes = torch.zeros(110, 10, 10)
        i_handle = linear(hxsnn.LIFObservables(spikes=spikes))
        y_handle = li(i_handle)

        self.assertIsNone(y_handle.membrane_cadc)
        self.assertIsNone(y_handle.membrane_madc)

        hxsnn.run(experiment, 110)

        # Assert types and shapes
        self.assertTrue(
            torch.equal(
                torch.tensor(y_handle.membrane_cadc.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(
            torch.equal(
                torch.tensor(y_handle.membrane_madc.shape),
                torch.tensor([2, 3235, 10])))

        # Only one module can record
        experiment = hxsnn.Experiment(dt=self.dt)
        experiment.calibration = calib_helper.fixture_calibration_from_file(
            calib_helper.nightly_calib_path()
        )

        linear_1 = hxsnn.Synapse(10, 10, experiment=experiment)
        li_1 = hxsnn.LI(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        linear_2 = hxsnn.Synapse(10, 10, experiment=experiment)
        li_2 = hxsnn.LI(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)

        spikes = torch.zeros(110, 10, 10)
        i_handle_1 = linear_1(hxsnn.LIFObservables(spikes=spikes))
        v_handle_1 = li_1(i_handle_1)
        i_handle_2 = linear_2(v_handle_1)
        li_2(i_handle_2)

        # Execute
        with self.assertRaises(ValueError):  # Expect ValueError
            hxsnn.run(experiment, 110)

    def test_events_on_membrane(self):
        """
        Test whether events arrive at desired membrane.
        """
        pass


if __name__ == "__main__":
    unittest.main()
