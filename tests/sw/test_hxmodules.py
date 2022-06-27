"""
Test HX Modules
"""
import unittest
import torch
from hxtorch.snn import Neuron, ReadoutNeuron, Synapse, Instance
from hxtorch.snn.snn import HXModule
from hxtorch.snn.handle import (
    TensorHandle, ReadoutNeuronHandle, NeuronHandle, SynapseHandle)


class TestHXModules(unittest.TestCase):
    """ Test HXModule """

    def test_is_autograd_function(self):
        """
        Test member _is_autograd_fn returns properly.
        """
        # No autograd function is used
        instance = Instance()
        module = HXModule(instance, lambda x: x)
        self.assertFalse(module._is_autograd_fn())

        # Autograd function is used
        instance = Instance()
        instance = Instance()
        module = HXModule(instance, torch.autograd.Function)
        self.assertTrue(module._is_autograd_fn())

    def test_forward(self):
        """
        Test Synapse returns the expected handle and registers module
        properly.
        """
        instance = Instance()
        module = HXModule(instance, None)
        # Test output handle
        input_handle = TensorHandle()
        synapse_handle = module(input_handle)
        self.assertTrue(isinstance(synapse_handle, TensorHandle))

        # Test module is registered in instance
        self.assertTrue(module in instance.connections.keys())
        # Test handles are assigned properly
        self.assertEqual(instance.connections[module][0][0], input_handle)
        self.assertEqual(instance.connections[module][1], synapse_handle)

    def test_prepare_function(self):
        """
        Test prepare_function strips func properly
        """
        # Test with normal function with params and hw_results
        def func(input, params=None, hw_data=None):
            return input, params, hw_data

        instance = Instance()
        module = HXModule(instance, func)
        module.extra_kwargs.update({"params": "new_params"})
        new_func = module.prepare_func("hw_result")
        output, params, result_ret = new_func(None)
        self.assertEqual(params, "new_params")
        self.assertEqual(result_ret, "hw_result")
        self.assertIsNone(output)

        # Test with normal function with params and without hw_results
        def func(input, params=None):
            return input, params

        instance = Instance()
        module = HXModule(instance, func)
        module.extra_kwargs.update({"params": "new_params"})
        new_func = module.prepare_func(None)
        output, params_ret = new_func(None)
        self.assertEqual(params_ret, "new_params")
        self.assertIsNone(output)

        # Test with normal function without params and with hw_results
        def func(input, hw_data=None):
            return input, hw_data

        instance = Instance()
        module = HXModule(instance, func)
        module.extra_kwargs.update({"params": "new_params"})
        new_func = module.prepare_func("hw_result")
        output, result_ret = new_func(None)
        self.assertEqual(result_ret, "hw_result")
        self.assertIsNone(output)

        # Test with autograd function
        class Func(torch.autograd.Function):
            def forward(ctx, input, params=None, hw_data=None):
                return input, params, hw_data

            def backward(ctx, grad):
                return grad

        instance = Instance()
        module = HXModule(instance, Func)
        module.extra_kwargs.update({"params": "new_params"})
        new_func = module.prepare_func("hw_result")
        output, params, result_ret = new_func(None)
        self.assertEqual(params, "new_params")
        self.assertEqual(result_ret, "hw_result")
        self.assertIsNone(output)

        # Test with autograd function, params and no hw_data
        class Func(torch.autograd.Function):
            def forward(ctx, input, params=None):
                return input, params

            def backward(ctx, grad):
                return grad

        instance = Instance()
        module = HXModule(instance, Func)
        module.extra_kwargs.update({"params": "new_params"})
        new_func = module.prepare_func("hw_result")
        output, params = new_func(None)
        self.assertEqual(params, "new_params")
        self.assertIsNone(output)

        # Test with autograd function, no params and hw_data
        class Func(torch.autograd.Function):
            def forward(ctx, input, hw_data=None):
                return input, hw_data

            def backward(ctx, grad):
                return grad

        instance = Instance()
        module = HXModule(instance, Func)
        new_func = module.prepare_func("hw_result")
        output, hw_result = new_func(None)
        self.assertEqual(hw_result, "hw_result")
        self.assertIsNone(output)

    def test_exec_forward(self):
        """
        Test execute_forward work as expected.
        """
        # Normal function
        def func(input, one, two, hw_data=None):
            self.assertEqual(hw_data, "hw_result")
            self.assertEqual((one, two), (1, 2))
            return input

        instance = Instance()
        module = HXModule(instance, func)
        module.extra_kwargs.update({"params": "new_params"})
        module.extra_args = (1, 2)

        # Input and output handles
        input_handle = NeuronHandle(torch.zeros(10, 5))
        output_handle = NeuronHandle()

        # Execute
        module.exec_forward(input_handle, output_handle, "hw_result")
        self.assertTrue(torch.equal(input_handle.spikes, output_handle.spikes))

        # Normal function
        def func(input, one, two, params=None, hw_data=None):
            self.assertEqual(params, "new_params")
            self.assertEqual(hw_data, "hw_result")
            self.assertEqual((one, two), (1, 2))
            return input

        instance = Instance()
        module = HXModule(instance, func)
        module.extra_kwargs.update({"params": "new_params"})
        module.extra_args = (1, 2)

        # Input and output handles
        input_handle = NeuronHandle(torch.zeros(10, 5))
        output_handle = NeuronHandle()

        # Execute
        module.exec_forward(input_handle, output_handle, "hw_result")
        self.assertTrue(torch.equal(input_handle.spikes, output_handle.spikes))

        # Autograd function
        class Func(torch.autograd.Function):
            def forward(ctx, input, one, two, params=None, hw_data=None):
                self.assertEqual(params, "new_params")
                self.assertEqual(hw_data, "hw_result")
                self.assertEqual((one, two), (1, 2))
                return input

            def backward(ctx, grad):
                return grad

        instance = Instance()
        module = HXModule(instance, Func)
        module.extra_kwargs.update({"params": "new_params"})
        module.extra_args = (1, 2)

        # Input and output handles
        input_handle = NeuronHandle(torch.zeros(10, 5))
        output_handle = NeuronHandle()

        # Execute
        module.exec_forward(input_handle, output_handle, "hw_result")
        self.assertTrue(torch.equal(input_handle.spikes, output_handle.spikes))

        # Autograd function no mock
        class Func(torch.autograd.Function):
            def forward(ctx, input, one, two, params=None, hw_data=None):
                self.assertEqual(params, "new_params")
                self.assertEqual(hw_data, "hw_result")
                self.assertEqual((one, two), (1, 2))
                return input

            def backward(ctx, grad):
                return grad

        instance = Instance(mock=False)
        module = HXModule(instance, Func)
        module.extra_kwargs.update({"params": "new_params"})
        module.extra_args = (1, 2)

        # Input and output handles
        input_handle = NeuronHandle(torch.zeros(10, 5))
        output_handle = NeuronHandle()

        # Execute
        module.exec_forward(input_handle, output_handle, "hw_result")
        # Hw result should be assigned here
        self.assertTrue(output_handle.spikes, "hw_result")


class TestSynapse(unittest.TestCase):
    """ Test Synapse """

    def test_output_type(self):
        """
        Test Synapse returns the expected handle
        """
        instance = Instance()
        synapse = Synapse(44, 33, instance)
        # Test output handle
        synapse_handle = synapse(NeuronHandle(spikes=torch.zeros(10, 44)))
        self.assertTrue(isinstance(synapse_handle, SynapseHandle))
        self.assertIsNone(synapse_handle.current)

    def test_weight_shape(self):
        """
        Test synapse weights are of correct shape.
        """
        instance = Instance()
        synapse = Synapse(44, 33, instance)
        # Test shape
        self.assertEqual(synapse.weight.shape[0], 33)
        self.assertEqual(synapse.weight.shape[1], 44)

    def test_weight_reset(self):
        """
        Test reset_parameters is working correctly
        """
        instance = Instance()
        synapse = Synapse(44, 33, instance)
        # Test weights are not zero (weight are initialized as zero and
        # reset_params is called implicitly)
        self.assertFalse(torch.equal(torch.zeros(33, 44), synapse.weight))


class TestNeuron(unittest.TestCase):
    """ Test Neuron """

    def test_output_type(self):
        """
        Test Neuron returns the expected handle
        """
        instance = Instance()
        neuron = Neuron(44, instance)
        # Test output handle
        neuron_handle = neuron(SynapseHandle(torch.zeros(10, 44)))
        self.assertTrue(isinstance(neuron_handle, NeuronHandle))
        self.assertIsNone(neuron_handle.spikes)
        self.assertIsNone(neuron_handle.membrane)


class TestReadoutNeuron(unittest.TestCase):
    """ Test ReadoutNeuron """

    def test_output_type(self):
        """
        Test ReadoutNeuron returns the expected handle
        """
        instance = Instance()
        neuron = ReadoutNeuron(44, instance)
        # Test output handle
        neuron_handle = neuron(SynapseHandle(torch.zeros(10, 44)))
        self.assertTrue(isinstance(neuron_handle, ReadoutNeuronHandle))
        self.assertIsNone(neuron_handle.membrane)


class TestDropout(unittest.TestCase):
    """ Test Neuron """

    def test_output_type(self):
        pass


if __name__ == "__main__":
    unittest.main()

