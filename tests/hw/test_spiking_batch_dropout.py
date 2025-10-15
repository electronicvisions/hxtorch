"""
Test snn.HXNeuron
"""
import unittest
from pathlib import Path

import torch

import hxtorch
from hxtorch import spiking as hxsnn

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


class TestBatchDropout(HWTestCase):
    """ Test hxtorch.snn.modules.BatchDropout """

    def test_output_type(self):
        """ Test BatchDropout returns expected handle """
        experiment = hxsnn.Experiment()
        dropout = hxsnn.BatchDropout(33, 0.5, experiment)
        # Test output handle
        dropout_handle = \
            dropout(hxsnn.LIFObservables(spikes=torch.zeros(10, 44)))
        self.assertTrue(isinstance(dropout_handle, hxsnn.LIFObservables))
        self.assertIsNone(dropout_handle.spikes)
        self.assertIsNone(dropout_handle.current)
        self.assertIsNone(dropout_handle.membrane_cadc)
        self.assertIsNone(dropout_handle.membrane_madc)

    def test_print(self):
        """ Test module printing """
        experiment = hxsnn.Experiment()
        module = hxsnn.BatchDropout(33, 0.5, experiment)
        logger.INFO(module)

    def test_set_mask(self):
        """ Test mask is updated properly """
        experiment = hxsnn.Experiment()
        dropout = hxsnn.BatchDropout(33, 0.5, experiment)

        # train mode
        dropout.train()
        mask1 = dropout.set_mask()
        self.assertTrue(torch.equal(mask1, dropout._mask))
        mask2 = dropout.set_mask()
        self.assertFalse(torch.equal(mask1, mask2))

        # eval mode
        dropout.eval()
        mask1 = dropout.set_mask()
        mask2 = dropout.set_mask()
        self.assertTrue(torch.equal(mask1, mask2))
        self.assertTrue(
            torch.equal(mask1, torch.ones_like(mask1)))

        # Test correct mask is passed to func
        class BatchDropout(hxsnn.BatchDropout):
            def forward_func(self, x):
                return x, self.mask

        experiment = hxsnn.Experiment()
        dropout = BatchDropout(33, 0.5, experiment)
        dropout.set_mask()
        input = torch.zeros(10, 10, 33)
        output, mask1 = dropout.func((input,))
        self.assertTrue(torch.equal(mask1, dropout._mask))
        self.assertTrue(torch.equal(output, input))
        new_mask = dropout.set_mask()
        input, mask2 = dropout.func((input,))
        self.assertTrue(torch.equal(mask2, dropout._mask))
        self.assertTrue(torch.equal(new_mask, mask2))


if __name__ == "__main__":
    unittest.main()
