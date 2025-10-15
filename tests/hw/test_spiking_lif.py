"""
Test snn.HXNeuron
"""
import unittest
from pathlib import Path

import torch

from dlens_vx_v3 import halco, lola

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


class TestLIF(HWTestCase):
    """ Test hxtorch.hxsnn.modules.LIF """

    def test_output_type(self):
        """
        Test neuron returns the expected handle
        """
        experiment = hxsnn.Experiment()
        neuron = hxsnn.LIF(44, experiment)
        # Test output handle
        neuron_handle = neuron(
            hxsnn.SynapseHandle(graded_spikes=torch.zeros(10, 44)))
        self.assertTrue(isinstance(neuron_handle, hxsnn.LIFObservables))
        self.assertIsNone(neuron_handle.spikes)
        self.assertIsNone(neuron_handle.membrane_cadc)
        self.assertIsNone(neuron_handle.membrane_madc)

    def test_print(self):
        """ Test module printing """
        experiment = hxsnn.Experiment()
        module = hxsnn.LIF(10, experiment=experiment)
        logger.INFO(module)

    def test_record_spikes(self):
        """
        Test spike recording with bypass mode.
        """
        # Enable bypass
        experiment = hxsnn.Experiment(dt=self.dt)
        experiment.calibration = calib_helper.fixture_calibration_from_chip(
            lola.Chip.default_neuron_bypass
        )

        # Modules
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        lif = hxsnn.LIF(
            10, enable_cadc_recording=True,  experiment=experiment)

        # Weights
        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63

        # Inputs
        spikes = torch.zeros(110, 10, 10)
        for idx in range(10):
            spikes[idx * 10 + 5, :, idx] = 1

        # Forward
        i_handle = linear(hxsnn.LIFObservables(spikes=spikes))
        s_handle = lif(i_handle)

        self.assertIsNone(s_handle.spikes)
        self.assertIsNone(s_handle.membrane_cadc)
        self.assertIsNone(s_handle.membrane_madc)

        # Execute
        hxsnn.run(experiment, 110)

        # Assert types and shapes
        self.assertIsInstance(s_handle.spikes, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(s_handle.membrane_cadc is not None)
        self.assertIsNone(s_handle.membrane_madc)

        # Assert data
        spike_times = torch.nonzero(s_handle.spikes)
        self.assertEqual(spike_times.shape[0], 10 * 10)

        i = 0
        for nrn in range(10):
            for b in range(10):
                self.assertEqual(b, spike_times[i, 1])
                # EA 2024-02-28: Sometimes spikes of first batch-entry seem to
                #                be delayed
                self.assertAlmostEqual(
                    5 + 10 * nrn, int(spike_times[i, 0]), delta=2)
                self.assertEqual(nrn, spike_times[i, 2])
                i += 1

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
        # Modules
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        lif = hxsnn.LIF(10, experiment=experiment)
        # Weights
        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63
        # Inputs
        spikes = torch.zeros(110, 10, 10)
        for idx in range(10):
            spikes[idx * 10 + 5, :, idx] = 1

        # Forward
        i_handle = linear(hxsnn.LIFObservables(spikes=spikes))
        s_handle = lif(i_handle)

        self.assertIsNone(s_handle.spikes)
        self.assertIsNone(s_handle.membrane_cadc)
        self.assertIsNone(s_handle.membrane_madc)

        # Execute
        hxsnn.run(experiment, 110)

        # Assert types and shapes
        self.assertIsInstance(s_handle.spikes, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.membrane_cadc.shape),
                torch.tensor([110, 10, 10])))
        self.assertIsNone(s_handle.membrane_madc)

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
        lif = hxsnn.LIF(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        spikes = torch.zeros(110, 10, 10)
        i_handle = linear(hxsnn.LIFObservables(spikes=spikes))
        s_handle = lif(i_handle)

        self.assertIsNone(s_handle.spikes)
        self.assertIsNone(s_handle.membrane_cadc)
        self.assertIsNone(s_handle.membrane_madc)

        hxsnn.run(experiment, 110)

        # Assert types and shapes
        self.assertIsInstance(s_handle.spikes, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.membrane_cadc.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.membrane_madc.shape),
                torch.tensor([2, 3235, 10])))
        # Only one module can record
        experiment = hxsnn.Experiment(dt=self.dt)
        experiment.calibration = calib_helper.fixture_calibration_from_file(
            calib_helper.nightly_calib_path()
        )
        linear_1 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif_1 = hxsnn.LIF(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        linear_2 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif_2 = hxsnn.LIF(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)

        spikes = torch.zeros(110, 10, 10)
        i_handle_1 = linear_1(hxsnn.LIFObservables(spikes=spikes))
        s_handle_1 = lif_1(i_handle_1)
        i_handle_2 = linear_2(s_handle_1)
        lif_2(i_handle_2)

        # Execute
        with self.assertRaises(ValueError):  # Expect ValueError
            hxsnn.run(experiment, 110)

    def test_events_on_membrane(self):
        """
        Test whether events arrive at desired membrane.
        """
        pass

    def test_neuron_spikes(self):
        """
        Test whether correct neuron does spike.
        """
        pass


if __name__ == "__main__":
    unittest.main()
