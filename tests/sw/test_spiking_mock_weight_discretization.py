"""
Test mock setting of the `Synapse` module.
"""
import unittest

import torch

import hxtorch.spiking as hxsnn


# pylint: disable=too-many-instance-attributes
class TestSynapseMockMode(unittest.TestCase):
    """
    Test weight discretization and saturation when using the mock setting of
    the `Synapse` module.
    """
    def test_synapse_mock_mode(self):
        """
        Test whether weight saturation and discretization work as intended when
        using the `Synapse` module with `mock = True`.
        """
        exp = hxsnn.Experiment(mock=True)
        spikes = torch.ones((1, 1, 1))
        weights = torch.nn.parameter.Parameter(
            torch.unsqueeze(torch.linspace(0, 100, 200), 1))
        weight_bounds = hxsnn.functional.mock.Bounds(
            lower=-63., upper=63.)
        synapse = hxsnn.modules.Synapse(1, 200, experiment=exp,
            mock=True, weight_step=1, weight_bounds=weight_bounds)
        synapse.weight.data = weights
        synapse_handle = synapse(
            hxsnn.LIFObservables(spikes=spikes))
        hxsnn.run(exp, 1)
        graded_spikes = synapse_handle.graded_spikes

        # Test for saturation
        self.assertTrue(torch.all(torch.le(
            graded_spikes, torch.ones(graded_spikes.shape) * 63.)))
        self.assertTrue(torch.all(torch.ge(
            graded_spikes, torch.ones(graded_spikes.shape) * (-63.))))
        # Test for discretization
        self.assertTrue(torch.allclose(
            graded_spikes, torch.round(graded_spikes)))


if __name__ == "__main__":
    unittest.main()
