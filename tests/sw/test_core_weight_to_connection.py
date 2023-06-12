"""
Test snn run function
"""
import unittest
import numpy as np
import _hxtorch_core


class TestWeightToConnection(unittest.TestCase):
    """ Test numpy weight matrix to grenade connection conversion """

    def test_weight_to_connection(self):
        """ Test 2-d weight array to grenade connections """
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

    def test_weight_list_to_connection(self):
        """ Test (weight, connections) to grenade connections """
        # test int
        weights = np.random.randint(0, 63, (5, 5)).reshape(-1)[:10]
        connections = np.random.randint(0, 5, (2, 10)).tolist()
        gconnections = _hxtorch_core.weight_to_connection(weights, connections)
        self.assertEqual(np.prod(weights.shape[0]), len(gconnections))
        for i, w in enumerate(weights):
            self.assertEqual(w, int(gconnections[i].weight))
            self.assertEqual(connections[0][i], int(gconnections[i].index_post[0]))
            self.assertEqual(connections[1][i], int(gconnections[i].index_pre[0]))

        # test negative int
        weights = -np.random.randint(0, 63, (5, 5)).reshape(-1)[:10]
        connections = np.random.randint(0, 5, (2, 10)).tolist()
        gconnections = _hxtorch_core.weight_to_connection(weights, connections)
        self.assertEqual(np.prod(weights.shape[0]), len(gconnections))
        for i, w in enumerate(weights):
            self.assertEqual(-w, int(gconnections[i].weight))
            self.assertEqual(connections[0][i], int(gconnections[i].index_post[0]))
            self.assertEqual(connections[1][i], int(gconnections[i].index_pre[0]))

        # test float
        weights = -np.random.randint(0, 63, (5, 5)).reshape(-1)[:10].astype(float)
        connections = np.random.randint(0, 5, (2, 10)).tolist()
        gconnections = _hxtorch_core.weight_to_connection(weights, connections)
        self.assertEqual(np.prod(weights.shape[0]), len(gconnections))
        for i, w in enumerate(weights):
            self.assertEqual(-w, int(gconnections[i].weight))
            self.assertEqual(connections[0][i], int(gconnections[i].index_post[0]))
            self.assertEqual(connections[1][i], int(gconnections[i].index_pre[0]))


if __name__ == "__main__":
    unittest.main()
