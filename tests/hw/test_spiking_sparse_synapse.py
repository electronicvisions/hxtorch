"""
Test snn.HXNeuron
"""
import unittest
from pathlib import Path

import torch

import hxtorch
from hxtorch import spiking as hxsnn
from hxtorch.core.utils import calib_helper


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


class TestSparseSynapse(HWTestCase):
    """ Test hxtorch.snn.modules.SparseSynapse """

    def test_output_type(self):
        """
        Test Synapse returns the expected handle
        """
        connections = (torch.randn(30, 44) < 0.1).float()
        experiment = hxsnn.Experiment()
        synapse = hxsnn.SparseSynapse(connections.to_sparse(), experiment)
        # Test output handle
        synapse_handle = synapse(
            hxsnn.LIFObservables(spikes=torch.zeros(10, 44)))
        self.assertTrue(isinstance(synapse_handle, hxsnn.SynapseHandle))
        self.assertIsNone(synapse_handle.graded_spikes)

    def test_weight_shape(self):
        """
        Test synapse weights are of correct shape.
        """
        connections = (torch.randn(30, 44) < 0.1).float()
        experiment = hxsnn.Experiment()
        synapse = hxsnn.SparseSynapse(connections, experiment)
        # Test shape
        self.assertEqual(synapse.weight.shape[0], 44)
        self.assertEqual(synapse.weight.shape[1], 30)

    def test_weight_reset(self):
        """
        Test reset_parameters is working correctly
        """
        connections = (torch.randn(30, 44) < 0.1).float()
        experiment = hxsnn.Experiment()
        synapse = hxsnn.SparseSynapse(connections, experiment)
        # Test weights are not zero (weight are initialized as zero and
        # reset_params is called implicitly)
        self.assertFalse(
            torch.equal(torch.zeros(44, 30), synapse.weight.to_dense()))

    def test_execution(self):
        """
        Test synapse is represented on hardware as expected
        """
        connections = (torch.randn(30, 44) < 0.1).float()
        experiment = hxsnn.Experiment(dt=self.dt)
        experiment.calibration = calib_helper.fixture_calibration_from_file(
            calib_helper.nightly_calib_path()
        )
        linear = hxsnn.SparseSynapse(
            connections.to_sparse(), experiment=experiment)
        lif = hxsnn.LI(44, experiment=experiment)
        spikes = torch.zeros(50, 1, 30)
        i_handle = linear(hxsnn.LIFObservables(spikes=spikes))
        lif(i_handle)
        hxsnn.run(experiment, 110)


if __name__ == "__main__":
    unittest.main()
