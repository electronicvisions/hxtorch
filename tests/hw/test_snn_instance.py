"""
Test HX Modules
"""
import unittest
import torch

import hxtorch
from hxtorch.snn import Instance
from hxtorch.snn.modules import HXModule, InputNeuron, Neuron, Synapse
from hxtorch.snn.handle import NeuronHandle


class TestInstance(unittest.TestCase):
    """ Test Instance """

    def setUp(cls):
        hxtorch.init_hardware(calib_name="spiking")

    def tearDown(cls):
        hxtorch.release_hardware()

    def test_connect(self):
        """
        Test connections are created correctly.
        """
        instance = Instance(mock=True)

        # Add one connection
        module1 = HXModule(instance, None)
        handle1 = NeuronHandle()
        handle2 = NeuronHandle()
        instance.connect(module1, handle1, handle2)

        # Check connection is registered
        sources = [
            e["handle"] for _, _, e in instance.modules.graph.in_edges(
                instance.modules.nodes[module1], data=True)]
        targets = [
            e["handle"] for _, _, e in instance.modules.graph.out_edges(
                instance.modules.nodes[module1], data=True)]
        self.assertEqual(len(sources), 1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(sources[0], handle1)
        self.assertEqual(targets[0], handle2)

        # Add another one
        module2 = HXModule(instance, None)
        handle3 = NeuronHandle()
        handle4 = NeuronHandle()
        instance.connect(module2, handle3, handle4)
        # Check connection is registered
        sources = [
            e["handle"] for _, _, e in instance.modules.graph.in_edges(
                instance.modules.nodes[module2], data=True)]
        targets = [
            e["handle"] for _, _, e in instance.modules.graph.out_edges(
                instance.modules.nodes[module2], data=True)]
        self.assertEqual(len(sources), 1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(sources[0], handle3)
        self.assertEqual(targets[0], handle4)

        # There should be two connections present now
        self.assertEqual(len(instance.modules.nodes), 2)

    def test_get_hw_result(self):
        """ Test hardware results are returned properly """
        # Mock mode
        instance = Instance(mock=True)
        # Modules
        module1 = Synapse(10, 10, instance, lambda x: x)
        module2 = Neuron(10, instance, lambda x: x)
        # Forward
        input_handle = NeuronHandle(spikes=torch.randn((10, 10, 10)))
        handle1 = module1(input_handle)
        module2(handle1)
        # Two modules should now be registered
        self.assertEqual(len(instance.modules.nodes), 2)
        # Get results -> In mock there are no hardware results
        results = instance.get_hw_results(10)
        self.assertEqual(results, dict())
        # No input node should be injected
        self.assertEqual(len(instance.modules.nodes), 2)
        # Do it again -> This should not change anything
        results = instance.get_hw_results(10)
        self.assertEqual(results, dict())
        self.assertEqual(len(instance.modules.nodes), 2)

        # HW mode
        instance = Instance(mock=False)
        # Modules
        module1 = Synapse(10, 10, instance, lambda x: x)
        module2 = Neuron(10, instance, lambda x: x)
        # Forward
        input_handle = NeuronHandle(spikes=torch.randn((10, 10, 10)))
        handle1 = module1(input_handle)
        module2(handle1)
        self.assertEqual(len(instance.modules.nodes), 2)
        results = instance.get_hw_results(10)
        self.assertEqual(len(instance.modules.nodes), 3)
        self.assertEqual(len(instance._populations), 2)
        self.assertEqual(len(instance._projections), 1)
        for pop in instance._populations:
            if isinstance(pop, InputNeuron):
                continue
            self.assertIsNotNone(results.get(pop.descriptor))

        # Execute again -> should still work as expected, in training we also
        results = instance.get_hw_results(10)
        self.assertEqual(len(instance.modules.nodes), 3)
        self.assertEqual(len(instance._populations), 2)
        self.assertEqual(len(instance._projections), 1)
        for pop in instance._populations:
            if isinstance(pop, InputNeuron):
                continue
            self.assertIsNotNone(results.get(pop.descriptor))

        # Deeper net
        instance = Instance(mock=False)
        # Modules
        module1 = Synapse(10, 10, instance, lambda x: x)
        module2 = Neuron(10, instance, lambda x: x)
        module3 = Synapse(10, 10, instance, lambda x: x)
        module4 = Neuron(10, instance, lambda x: x)
        module5 = Synapse(10, 10, instance, lambda x: x)
        module6 = Neuron(10, instance, lambda x: x)

        # Forward
        input_handle = NeuronHandle(spikes=torch.randn((20, 10, 10)))
        handle1 = module1(input_handle)
        handle2 = module2(handle1)
        handle3 = module3(handle2)
        handle4 = module4(handle3)
        handle5 = module5(handle4)
        module6(handle5)

        # Six modules should now be registered
        self.assertEqual(len(instance.modules.nodes), 6)
        results = instance.get_hw_results(20)
        self.assertEqual(len(instance.modules.nodes), 7)
        self.assertEqual(len(instance._populations), 4)
        self.assertEqual(len(instance._projections), 3)
        for pop in instance._populations:
            if isinstance(pop, InputNeuron):
                continue
            self.assertIsNotNone(results.get(pop.descriptor))


if __name__ == "__main__":
    unittest.main()
