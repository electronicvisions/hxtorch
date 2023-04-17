"""
Test snn run function
"""
import unittest
import numpy as np
import _hxtorch_core


class TestDenseSpikesToList(unittest.TestCase):
    """Test numpy dense spike to grenade spike list conversion"""

    def test_dense_spikes_to_list(self):
        """Test run in abstract case """
        # test int
        spike_idx = np.array([[0, 2, 1, 3, 1, 1]])
        spike_time = np.array([[0.1, 0.2, 0.5, 1.0, 1.2, 1.5]])
        spikes = (spike_idx, spike_time)
        grenade_format = [[0.1], [0.5, 1.2, 1.5], [0.2], [1.0]]

        input_neurons = 4
        spike_list = _hxtorch_core.dense_spikes_to_list(spikes, input_neurons)
        self.assertEqual(len(spike_list), 1)
        first_batch = spike_list[0]

        for a, b in zip(first_batch, grenade_format):
            for c, d in zip(a, b):
                self.assertAlmostEqual(c, d)


if __name__ == "__main__":
    unittest.main()
