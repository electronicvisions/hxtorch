"""
Test snn run function
"""
import unittest
import torch
from hxtorch.spiking import run, Experiment
from hxtorch.spiking.modules import Neuron, Synapse
from hxtorch.spiking.handle import NeuronHandle


class TestSNNRun(unittest.TestCase):
    """ Test snn run """

    def test_run(self):
        """ Test run in abstract case """
        # Experiment
        experiment = Experiment(mock=True)

        # Modules
        module1 = Neuron(10, experiment, lambda x: x)
        module2 = Neuron(10, experiment, lambda x: x)
        module3 = Neuron(10, experiment, lambda x: x)

        # Input handle
        input_handle = NeuronHandle(torch.zeros(10, 12))
        h1 = module1(input_handle)
        h2 = module2(h1)
        h3 = module3(h2)

        # Handles should be empty
        self.assertIsNone(h1.observable_state)
        self.assertIsNone(h2.observable_state)
        self.assertIsNone(h3.observable_state)

        # Run
        run(experiment, None)

        # Handles should be full now
        self.assertTrue(torch.equal(h1.observable_state, torch.zeros(10, 12)))
        self.assertTrue(torch.equal(h2.observable_state, torch.zeros(10, 12)))
        self.assertTrue(torch.equal(h3.observable_state, torch.zeros(10, 12)))

    def test_run_realistic(self):
        """ Test run in realistiv scenario """
        # Experiment
        experiment = Experiment(mock=True)

        def syn_func(x, w, b):
            return x

        # Modules
        l1 = Synapse(5, 10, experiment, syn_func)
        n1 = Neuron(10, experiment, lambda x: x)
        l2 = Synapse(10, 20, experiment, syn_func)
        n2 = Neuron(20, experiment, lambda x: x)
        l3 = Synapse(20, 1, experiment, syn_func)
        n3 = Neuron(1, experiment, lambda x: x)

        # Input handle
        input_handle = NeuronHandle(torch.zeros(10, 5))
        h1 = l1(input_handle)
        h2 = n1(h1)
        h3 = l2(h2)
        h4 = n2(h3)
        h5 = l3(h4)
        h6 = n3(h5)

        # Handles should be empty
        self.assertIsNone(h1.observable_state)
        self.assertIsNone(h2.observable_state)
        self.assertIsNone(h3.observable_state)
        self.assertIsNone(h4.observable_state)
        self.assertIsNone(h5.observable_state)
        self.assertIsNone(h6.observable_state)

        # Run
        run(experiment, None)

        # Handles should be full now
        self.assertTrue(torch.equal(h1.observable_state, torch.zeros(10, 5)))
        self.assertTrue(torch.equal(h2.observable_state, torch.zeros(10, 5)))
        self.assertTrue(torch.equal(h3.observable_state, torch.zeros(10, 5)))
        self.assertTrue(torch.equal(h4.observable_state, torch.zeros(10, 5)))
        self.assertTrue(torch.equal(h5.observable_state, torch.zeros(10, 5)))
        self.assertTrue(torch.equal(h6.observable_state, torch.zeros(10, 5)))


if __name__ == "__main__":
    unittest.main()
