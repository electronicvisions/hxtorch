"""
Test HX Modules
"""
import unittest
from collections import OrderedDict
from hxtorch.snn import Instance
from hxtorch.snn.snn import HXModule
from hxtorch.snn.handle import NeuronHandle


class TestInstance(unittest.TestCase):
    """ Test Instance """

    def test_connect(self):
        """
        Test connections are created correctly.
        """
        instance = Instance()

        # Add one connection
        module1 = HXModule(instance, None)
        input_handle1 = NeuronHandle()
        output_handle1 = NeuronHandle()
        instance.connect(module1, input_handle1, output_handle1)

        # Check connection is registered
        self.assertEqual(
            instance.connections[module1], (input_handle1, output_handle1))

        # Add another one
        module2 = HXModule(instance, None)
        input_handle2 = NeuronHandle()
        output_handle2 = NeuronHandle()
        instance.connect(module2, input_handle2, output_handle2)

        # Check connection is registered
        self.assertEqual(
            instance.connections[module2], (input_handle2, output_handle2))

        # There should be two connections present now
        self.assertEqual(len(instance.connections), 2)

    def test_sort(self):
        """
        Test connections are sorted correctly.
        """
        # Test correct
        instance = Instance()

        # Modules
        module1 = HXModule(instance, None)
        module2 = HXModule(instance, None)
        module3 = HXModule(instance, None)

        # Handles
        handle1 = NeuronHandle()
        handle2 = NeuronHandle()
        handle3 = NeuronHandle()
        handle4 = NeuronHandle()

        # Add connections in wrong order
        instance.connect(module1, handle1, handle2)
        instance.connect(module2, handle2, handle3)
        instance.connect(module3, handle3, handle4)

        # Sort
        target_modules = OrderedDict()
        target_modules[module1] = (handle1, handle2)
        target_modules[module2] = (handle2, handle3)
        target_modules[module3] = (handle3, handle4)
        sorted_modules = instance.sorted()

        target_keys = target_modules.keys()
        sorted_keys = sorted_modules.keys()
        target_values = target_modules.values()
        sorted_values = sorted_modules.values()

        for m1, m2 in zip(target_keys, sorted_keys):
            self.assertEqual(m1, m2)
        for h1, h2 in zip(target_values, sorted_values):
            self.assertEqual(h1, h2)

        # Test wrong order
        instance = Instance()

        # Modules
        module1 = HXModule(instance, None)
        module2 = HXModule(instance, None)
        module3 = HXModule(instance, None)

        # Handles
        handle1 = NeuronHandle()
        handle2 = NeuronHandle()
        handle3 = NeuronHandle()
        handle4 = NeuronHandle()

        # Add connections in wrong order
        instance.connect(module2, handle2, handle3)
        instance.connect(module1, handle1, handle2)
        instance.connect(module3, handle3, handle4)

        # Sort
        target_modules = OrderedDict()
        target_modules[module1] = (handle1, handle2)
        target_modules[module2] = (handle2, handle3)
        target_modules[module3] = (handle3, handle4)
        sorted_modules = instance.sorted()

        target_keys = target_modules.keys()
        sorted_keys = sorted_modules.keys()
        target_values = target_modules.values()
        sorted_values = sorted_modules.values()

        for m1, m2 in zip(target_keys, sorted_keys):
            self.assertEqual(m1, m2)
        for h1, h2 in zip(target_values, sorted_values):
            self.assertEqual(h1, h2)

        # Test independet sub-graphes
        instance = Instance()

        # Modules
        module1 = HXModule(instance, None)
        module2 = HXModule(instance, None)
        module3 = HXModule(instance, None)
        module4 = HXModule(instance, None)

        # Handles
        handle1 = NeuronHandle()
        handle2 = NeuronHandle()
        handle3 = NeuronHandle()
        handle4 = NeuronHandle()
        handle5 = NeuronHandle()
        handle6 = NeuronHandle()

        # Add connections in wrong order
        instance.connect(module1, handle1, handle2)
        instance.connect(module2, handle2, handle3)
        instance.connect(module4, handle5, handle6)
        instance.connect(module3, handle4, handle5)

        # Sort
        target_modules = OrderedDict()
        target_modules[module1] = (handle1, handle2)
        target_modules[module2] = (handle2, handle3)
        target_modules[module3] = (handle4, handle5)
        target_modules[module4] = (handle5, handle6)
        sorted_modules = instance.sorted()

        target_keys = target_modules.keys()
        sorted_keys = sorted_modules.keys()
        target_values = target_modules.values()
        sorted_values = sorted_modules.values()

        for m1, m2 in zip(target_keys, sorted_keys):
            self.assertEqual(m1, m2)
        for h1, h2 in zip(target_values, sorted_values):
            self.assertEqual(h1, h2)

        # Test loop
        instance = Instance()

        # Modules
        module1 = HXModule(instance, None)
        module2 = HXModule(instance, None)
        module3 = HXModule(instance, None)

        # Handles
        handle1 = NeuronHandle()
        handle2 = NeuronHandle()
        handle3 = NeuronHandle()
        handle4 = NeuronHandle()
        handle5 = NeuronHandle()
        handle6 = NeuronHandle()

        # Add connections in wrong order
        instance.connect(module1, (handle1, handle6), handle2)
        instance.connect(module2, handle2, handle3)
        instance.connect(module3, handle3, handle1)

        # Sort
        target_modules = OrderedDict()
        target_modules[module1] = ((handle1, handle6), handle2)
        target_modules[module2] = (handle2, handle3)
        target_modules[module3] = (handle3, handle1)
        sorted_modules = instance.sorted()

        target_keys = target_modules.keys()
        sorted_keys = sorted_modules.keys()
        target_values = target_modules.values()
        sorted_values = sorted_modules.values()

        for m1, m2 in zip(target_keys, sorted_keys):
            self.assertEqual(m1, m2)
        for h1, h2 in zip(target_values, sorted_values):
            self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
