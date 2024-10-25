"""
Test snn run function
"""
import unittest
import torch
from hxtorch.spiking import run, Experiment
from hxtorch.spiking.modules import HXModule, Neuron, Synapse
from hxtorch.spiking.handle import TensorHandle, NeuronHandle


class TestSNNRun(unittest.TestCase):
    """ Test snn run """

    def test_run(self):
        """ Test run in abstract case """
        # Experiment
        experiment = Experiment(mock=True)

        # Modules
        module1 = HXModule(experiment)
        module2 = HXModule(experiment)
        module3 = HXModule(experiment)

        # Input handle
        print("Experiment: ", experiment, module1.experiment)
        input_handle = TensorHandle(torch.zeros(10, 1, 12))
        h1 = module1(input_handle)
        h2 = module2(h1)
        h3 = module3(h2)

        # Handles should be empty
        self.assertIsNone(h1.tensor)
        self.assertIsNone(h2.tensor)
        self.assertIsNone(h3.tensor)

        # Run
        run(experiment, None)

        # Handles should be full now
        self.assertTrue(torch.equal(h1.tensor, torch.zeros(10, 1, 12)))
        self.assertTrue(torch.equal(h2.tensor, torch.zeros(10, 1, 12)))
        self.assertTrue(torch.equal(h3.tensor, torch.zeros(10, 1, 12)))

    def test_run_realistic(self):
        """ Test run in realistic scenario """
        # Experiment
        experiment = Experiment(mock=True)

        # Modules
        l1 = Synapse(5, 10, experiment)
        n1 = Neuron(10, experiment)
        l2 = Synapse(10, 20, experiment)
        n2 = Neuron(20, experiment)
        l3 = Synapse(20, 1, experiment)
        n3 = Neuron(1, experiment)

        # Input handle
        input_handle = NeuronHandle(torch.zeros(10, 1, 5))
        h1 = l1(input_handle)
        h2 = n1(h1)
        h3 = l2(h2)
        h4 = n2(h3)
        h5 = l3(h4)
        h6 = n3(h5)

        # Handles should be empty
        self.assertIsNone(h1.graded_spikes)
        self.assertIsNone(h2.spikes)
        self.assertIsNone(h3.graded_spikes)
        self.assertIsNone(h4.spikes)
        self.assertIsNone(h5.graded_spikes)
        self.assertIsNone(h6.spikes)

        # Run
        run(experiment, None)

        # Handles should be full now
        self.assertTrue(torch.equal(h1.graded_spikes, torch.zeros(10, 1, 10)))
        self.assertTrue(torch.equal(h2.spikes, torch.zeros(10, 1, 10)))
        self.assertTrue(torch.equal(h3.graded_spikes, torch.zeros(10, 1, 20)))
        self.assertTrue(torch.equal(h4.spikes, torch.zeros(10, 1, 20)))
        self.assertTrue(torch.equal(h5.graded_spikes, torch.zeros(10, 1, 1)))
        self.assertTrue(torch.equal(h6.spikes, torch.zeros(10, 1, 1)))


if __name__ == "__main__":
    unittest.main()
