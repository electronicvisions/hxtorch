"""
Test snn.HXNeuron
"""
import unittest
from pathlib import Path

import torch

import hxtorch
from hxtorch import spiking as hxsnn


hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)
logger = hxtorch.logger.get("hxtorch.test.hw.test_spiking_synapse")


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


class TestSynapse(HWTestCase):
    """ Test hxtorch.snn.modules.Synapse """

    def test_output_type(self):
        """
        Test Synapse returns the expected handle
        """
        experiment = hxsnn.Experiment()
        synapse = hxsnn.Synapse(44, 33, experiment)
        # Test output handle
        synapse_handle = \
            synapse(hxsnn.LIFObservables(spikes=torch.zeros(10, 44)))
        self.assertTrue(isinstance(synapse_handle, hxsnn.SynapseHandle))
        self.assertIsNone(synapse_handle.graded_spikes)

    def test_print(self):
        """ Test module printing """
        experiment = hxsnn.Experiment()
        module = hxsnn.Synapse(10, 22, experiment=experiment)
        logger.INFO(module)

    def test_weight_shape(self):
        """
        Test synapse weights are of correct shape.
        """
        experiment = hxsnn.Experiment()
        synapse = hxsnn.Synapse(44, 33, experiment)
        # Test shape
        self.assertEqual(synapse.weight.shape[0], 33)
        self.assertEqual(synapse.weight.shape[1], 44)

    def test_weight_reset(self):
        """
        Test reset_parameters is working correctly
        """
        experiment = hxsnn.Experiment()
        synapse = hxsnn.Synapse(44, 33, experiment)
        # Test weights are not zero (weight are initialized as zero and
        # reset_params is called implicitly)
        self.assertFalse(torch.equal(torch.zeros(33, 44), synapse.weight))

    def test_signed_projection(self):
        """
        Test synapse is represented on hardware as expected
        """
        pass


if __name__ == "__main__":
    unittest.main()
