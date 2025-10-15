"""
Test spike extraction utilities.

Tests _hxtorch_core.extract_n_spikes and _hxtorch_core.dense_spikes_to_list.
"""
import unittest

import numpy as np
import _hxtorch_core


class TestExtractNSpikes(unittest.TestCase):
    """Test extract_n_spikes with spike data."""

    batch_size: int = 3
    pop_size: int = 3

    def test_extract_n_spikes_synthetic(self):
        """Test extract_n_spikes with known spike time data."""
        spike_idx = np.array(
            [[0, 1, 2, 0, 1], [0, 2, 1, 0, 2], [1, 0, 2, 1, 0]])
        spike_time = np.array(
            [[0.1, 0.2, 0.3, 0.5, 0.6],
             [0.1, 0.2, 0.3, 0.5, 0.6],
             [0.1, 0.2, 0.3, 0.5, 0.6]])
        spikes = (spike_idx, spike_time)
        spike_list = _hxtorch_core.dense_spikes_to_list(
            spikes, self.pop_size)

        self.assertEqual(len(spike_list), self.batch_size)

        # Extract n spikes
        n_events = 4
        max_spikes = 2
        indices, times = _hxtorch_core.extract_n_spikes(
            spike_list, n_events, max_spikes)

        self.assertEqual(list(indices.shape), [self.batch_size, n_events])
        self.assertEqual(list(times.shape), [self.batch_size, n_events])

        # Unfilled entries should be -1 (indices) and inf (times)
        for b in range(self.batch_size):
            for e in range(n_events):
                if indices[b, e] == -1:
                    self.assertEqual(times[b, e], np.inf)


if __name__ == "__main__":
    unittest.main()
