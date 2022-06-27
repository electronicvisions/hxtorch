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
        hxtorch.init_hardware(spiking=True)

    def tearDown(cls):
        hxtorch.release_hardware()

    def test_connect(self):
        """
        Test connections are created correctly.
        """
        instance = Instance(mock=True)

        # Add one connection
        module1 = HXModule(instance, None)
        input_handle1 = NeuronHandle()
        output_handle1 = NeuronHandle()
        instance.connect(module1, input_handle1, output_handle1)

        # Check connection is registered
        self.assertEqual(
            instance.modules.get_node(module1).input_handle, (input_handle1,))
        self.assertEqual(
            instance.modules.get_node(module1).output_handle, output_handle1)

        # Add another one
        module2 = HXModule(instance, None)
        input_handle2 = NeuronHandle()
        output_handle2 = NeuronHandle()
        instance.connect(module2, input_handle2, output_handle2)

        # Check connection is registered
        self.assertEqual(
            instance.modules.get_node(module2).input_handle,
            (input_handle2,))
        self.assertEqual(
            instance.modules.get_node(module2).output_handle, output_handle2)

        # There should be two connections present now
        self.assertEqual(len(instance.modules), 2)

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
        self.assertEqual(len(instance.modules), 2)
        # Get results -> In mock there are no hardware results
        results = instance.get_hw_results(10)
        self.assertEqual(results, dict())
        self.assertEqual(len(instance.modules), 2)
        # Do it again -> This should not change anything
        results = instance.get_hw_results(10)
        self.assertEqual(results, dict())
        self.assertEqual(len(instance.modules), 2)

        # HW mode
        instance = Instance(mock=False)
        # Modules
        module1 = Synapse(10, 10, instance, lambda x: x)
        module2 = Neuron(10, instance, lambda x: x)
        # Forward
        input_handle = NeuronHandle(spikes=torch.randn((10, 10, 10)))
        handle1 = module1(input_handle)
        module2(handle1)
        # Two modules should now be registered
        self.assertEqual(len(instance.modules), 2)
        # Get results -> In mock there are no hardware results
        results = instance.get_hw_results(10)
        print("First")
        # Test
        self.assertEqual(len(instance.modules), 3)  # InputNeuron added
        self.assertEqual(len(instance.modules.populations), 2)
        self.assertEqual(len(instance.modules.projections), 1)
        for pop in instance.modules.populations:
            if isinstance(pop.module, InputNeuron):
                continue
            self.assertIsNotNone(results.get(pop.descriptor))

        # Forward again -> should still work as expected, in training we also
        # loop
        input_handle = NeuronHandle(spikes=torch.randn((10, 10, 10)))
        handle1 = module1(input_handle)
        module2(handle1)
        self.assertEqual(len(instance.modules), 3)  # InputNeuron still added
        # Get results -> In mock there are no hardware results
        print("Second")
        print(instance.modules)
        results = instance.get_hw_results(10)
        print("Second")
        # Test
        self.assertEqual(len(instance.modules), 3)  # InputNeuron still added
        self.assertEqual(len(instance.modules.populations), 2)
        self.assertEqual(len(instance.modules.projections), 1)
        for pop in instance.modules.populations:
            if isinstance(pop.module, InputNeuron):
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
        input_handle = NeuronHandle(spikes=torch.randn((10, 20, 10)))
        handle1 = module1(input_handle)
        handle2 = module2(handle1)
        handle3 = module3(handle2)
        handle4 = module4(handle3)
        handle5 = module5(handle4)
        module6(handle5)
        print(instance.modules)

        # Two modules should now be registered
        self.assertEqual(len(instance.modules), 6)
        # Get results -> In mock there are no hardware results
        results = instance.get_hw_results(10)
        # Test
        self.assertEqual(len(instance.modules), 7)  # InputNeuron added
        self.assertEqual(len(instance.modules.populations), 4)
        self.assertEqual(len(instance.modules.projections), 3)
        for pop in instance.modules.populations:
            if isinstance(pop.module, InputNeuron):
                continue
            self.assertIsNotNone(results.get(pop.descriptor))


if __name__ == "__main__":
    unittest.main()
