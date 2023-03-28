"""
Test snn run function
"""
import unittest
import numpy as np
import _hxtorch_core


class TestWeightToConnection(unittest.TestCase):
    """ Test numpy weight matrix to grenade connection conversion """

    def test_weight_to_connection(self):
        """ Test run in abstract case """
        # test int
        weights = np.random.randint(0, 63, (5, 5))
        connections = _hxtorch_core.weight_to_connection(weights)
        self.assertEqual(np.prod(weights.shape), len(connections))
        c = 0
        for i, col in enumerate(weights):
            for k, w in enumerate(col):
                self.assertEqual(w, int(connections[c].weight))
                c += 1

        # test negativ int
        weights = -np.random.randint(0, 63, (5, 5))
        connections = _hxtorch_core.weight_to_connection(weights)
        self.assertEqual(np.prod(weights.shape), len(connections))
        c = 0
        for i, col in enumerate(weights):
            for k, w in enumerate(col):
                self.assertEqual(-w, int(connections[c].weight))
                c += 1

        # test float
        weights = np.random.randint(0, 63, (5, 5)).astype(float)
        connections = _hxtorch_core.weight_to_connection(weights)
        self.assertEqual(np.prod(weights.shape), len(connections))
        c = 0
        for i, col in enumerate(weights):
            for k, w in enumerate(col):
                self.assertEqual(w, int(connections[c].weight))
                c += 1


if __name__ == "__main__":
    unittest.main()
