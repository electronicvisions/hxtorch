"""
Test snn.HXNeuron
"""
import unittest
import torch
from dlens_vx_v3 import lola
import hxtorch
from hxtorch import snn

hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)


class TestHXModules(unittest.TestCase):
    """ Test HXModule """

    def test_is_autograd_function(self):
        """
        Test member _is_autograd_fn returns properly.
        """
        # No autograd function is used
        instance = snn.Instance(mock=True)
        module = snn.HXModule(instance, lambda x: x)
        self.assertFalse(module._is_autograd_fn())

        # Autograd function is used
        instance = snn.Instance(mock=True)
        module = snn.HXModule(instance, torch.autograd.Function)
        self.assertTrue(module._is_autograd_fn())

    def test_forward(self):
        """
        Test Synapse returns the expected handle and registers module
        properly.
        """
        instance = snn.Instance(mock=True)
        module = snn.HXModule(instance, None)
        # Test output handle
        input_handle = snn.TensorHandle()
        synapse_handle = module(input_handle)
        self.assertTrue(isinstance(synapse_handle, snn.TensorHandle))

        # Test module is registered in instance
        self.assertTrue(instance.modules.module_exists(module))
        # Test handles are assigned properly
        self.assertEqual(
            instance.modules.get_node(module).input_handle[0], input_handle)
        self.assertEqual(
            instance.modules.get_node(module).output_handle, synapse_handle)

    def test_prepare_function(self):
        """
        Test prepare_function strips func properly
        """
        # Test with normal function with params and hw_results
        def func(input, params=None, hw_data=None):
            return input, params, hw_data

        instance = snn.Instance(mock=True)
        module = snn.HXModule(instance, func)
        module.extra_kwargs.update({"params": "new_params"})
        new_func = module.prepare_func("hw_result")
        output, params, result_ret = new_func(None)
        self.assertEqual(params, "new_params")
        self.assertEqual(result_ret, "hw_result")
        self.assertIsNone(output)

        # Test with normal function with params and without hw_results
        def func(input, params=None):
            return input, params

        instance = snn.Instance(mock=True)
        module = snn.HXModule(instance, func)
        module.extra_kwargs.update({"params": "new_params"})
        new_func = module.prepare_func(None)
        output, params_ret = new_func(None)
        self.assertEqual(params_ret, "new_params")
        self.assertIsNone(output)

        # Test with normal function without params and with hw_results
        def func(input, hw_data=None):
            return input, hw_data

        instance = snn.Instance(mock=True)
        module = snn.HXModule(instance, func)
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

        instance = snn.Instance(mock=True)
        module = snn.HXModule(instance, Func)
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

        instance = snn.Instance(mock=True)
        module = snn.HXModule(instance, Func)
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

        instance = snn.Instance(mock=True)
        module = snn.HXModule(instance, Func)
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

        instance = snn.Instance(mock=True)
        module = snn.HXModule(instance, func)
        module.extra_kwargs.update({"params": "new_params"})
        module.extra_args = (1, 2)

        # Input and output handles
        input_handle = snn.NeuronHandle(torch.zeros(10, 5))
        output_handle = snn.NeuronHandle()

        # Execute
        module.exec_forward(input_handle, output_handle, "hw_result")
        self.assertTrue(torch.equal(input_handle.spikes, output_handle.spikes))

        # Normal function
        def func(input, one, two, params=None, hw_data=None):
            self.assertEqual(params, "new_params")
            self.assertEqual(hw_data, "hw_result")
            self.assertEqual((one, two), (1, 2))
            return input

        instance = snn.Instance(mock=True)
        module = snn.HXModule(instance, func)
        module.extra_kwargs.update({"params": "new_params"})
        module.extra_args = (1, 2)

        # Input and output handles
        input_handle = snn.NeuronHandle(torch.zeros(10, 5))
        output_handle = snn.NeuronHandle()

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

        instance = snn.Instance(mock=True)
        module = snn.HXModule(instance, Func)
        module.extra_kwargs.update({"params": "new_params"})
        module.extra_args = (1, 2)

        # Input and output handles
        input_handle = snn.NeuronHandle(torch.zeros(10, 5))
        output_handle = snn.NeuronHandle()

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

        instance = snn.Instance(mock=False)
        module = snn.HXModule(instance, Func)
        module.extra_kwargs.update({"params": "new_params"})
        module.extra_args = (1, 2)

        # Input and output handles
        input_handle = snn.NeuronHandle(torch.zeros(10, 5))
        output_handle = snn.NeuronHandle()

        # Execute
        module.exec_forward(input_handle, output_handle, "hw_result")
        # Hw result should be assigned here
        self.assertTrue(output_handle.spikes, "hw_result")


class HWTestCase(unittest.TestCase):
    """ HW setup """

    dt = 1.0e-6

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware(spiking=True)

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()


class TestNeuron(HWTestCase):
    """ Test hxtorch.snn.modules.Neuron """

    def test_output_type(self):
        """
        Test neuron returns the expected handle
        """
        instance = snn.Instance()
        neuron = snn.Neuron(44, instance)
        # Test output handle
        neuron_handle = neuron(snn.SynapseHandle(torch.zeros(10, 44)))
        self.assertTrue(isinstance(neuron_handle, snn.NeuronHandle))
        self.assertIsNone(neuron_handle.spikes)
        self.assertIsNone(neuron_handle.v_cadc)
        self.assertIsNone(neuron_handle.v_madc)

    def test_record_spikes(self):
        """
        Test spike recording with bypass mode.
        """
        # Enable bypass
        instance = snn.Instance(dt=self.dt)
        instance.initial_config = lola.Chip.default_neuron_bypass

        # Modules
        linear = snn.Synapse(10, 10, instance=instance)
        lif = snn.Neuron(10, enable_cadc_recording=False,  instance=instance)

        # Weights
        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63

        # Inputs
        spikes = torch.zeros(10, 110, 10)
        for idx in range(10):
            spikes[:, idx * 10 + 5, idx] = 1

        # Forward
        i_handle = linear(snn.NeuronHandle(spikes))
        s_handle = lif(i_handle)

        self.assertTrue(s_handle.spikes is None)
        self.assertTrue(s_handle.v_cadc is None)
        self.assertTrue(s_handle.v_madc is None)

        # Execute
        snn.run(instance, 110)

        # Assert types and shapes
        self.assertIsInstance(s_handle.spikes, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([10, 110 + 1, 10])))
        self.assertTrue(s_handle.v_cadc is None)
        self.assertTrue(s_handle.v_madc is None)

        # Assert data
        spike_times = torch.nonzero(s_handle.spikes)
        self.assertEqual(spike_times.shape[0], 10 * 10)

        i = 0
        for b in range(10):
            for nrn in range(10):
                self.assertEqual(b, spike_times[i, 0])
                self.assertEqual(5 + 10 * nrn, spike_times[i, 1])
                self.assertEqual(nrn, spike_times[i, 2])
                i += 1

    def test_record_cadc(self):
        """
        Test CADC recording.

        TODO:
            - Ensure correct order.
        """
        instance = snn.Instance(dt=self.dt)

        # Modules
        linear = snn.Synapse(10, 10, instance=instance)
        lif = snn.Neuron(10, instance=instance)

        # Weights
        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63

        # Inputs
        spikes = torch.zeros(10, 110, 10)
        for idx in range(10):
            spikes[:, idx * 10 + 5, idx] = 1

        # Forward
        i_handle = linear(snn.NeuronHandle(spikes))
        s_handle = lif(i_handle)

        self.assertTrue(s_handle.spikes is None)
        self.assertTrue(s_handle.v_cadc is None)
        self.assertTrue(s_handle.v_madc is None)

        # Execute
        snn.run(instance, 110)

        # Assert types and shapes
        self.assertIsInstance(s_handle.spikes, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([10, 110 + 1, 10])))
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.v_cadc.shape),
                torch.tensor([10, 110 + 1, 10])))
        self.assertTrue(s_handle.v_madc is None)

    def test_record_madc(self):
        """
        Test MADC recording.

        TODO:
            - Ensure correct neuron is recorded.
        """
        instance = snn.Instance(dt=self.dt)
        linear = snn.Synapse(10, 10, instance=instance)
        lif = snn.Neuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            instance=instance)

        spikes = torch.zeros(10, 110, 10)
        i_handle = linear(snn.NeuronHandle(spikes))
        lif(i_handle)

        # TODO: Adjust as soon `to_dense` for MADC samples is implemented.
        with self.assertRaises(NotImplementedError):
            snn.run(instance, 110)

        # Only one module can record
        instance = snn.Instance(dt=self.dt)
        linear_1 = snn.Synapse(10, 10, instance=instance)
        lif_1 = snn.Neuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            instance=instance)
        linear_2 = snn.Synapse(10, 10, instance=instance)
        lif_2 = snn.Neuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            instance=instance)

        spikes = torch.zeros(10, 110, 10)
        i_handle_1 = linear_1(snn.NeuronHandle(spikes))
        s_handle_1 = lif_1(i_handle_1)
        i_handle_2 = linear_2(s_handle_1)
        lif_2(i_handle_2)

        # Execute
        with self.assertRaises(RuntimeError):  # Expect RuntimeError
            snn.run(instance, 110)

    def test_events_on_membrane(self):
        """
        Test whether events arrive at desired membrane.
        """
        pass

    def test_neuron_spikes(self):
        """
        Test whether correct neuron does spike.
        """
        pass


class TestReadoutNeuron(HWTestCase):
    """ Test hxtorch.snn.modules.ReadoutNeuron """

    def test_output_type(self):
        """
        Test ReadoutNeuron returns the expected handle
        """
        instance = snn.Instance()
        neuron = snn.ReadoutNeuron(44, instance)
        # Test output handle
        neuron_handle = neuron(snn.SynapseHandle(torch.zeros(10, 44)))
        self.assertTrue(isinstance(neuron_handle, snn.ReadoutNeuronHandle))
        self.assertIsNone(neuron_handle.v_cadc)
        self.assertIsNone(neuron_handle.v_madc)

    def test_record_cadc(self):
        """
        Test CADC recording.

        TODO:
            - Ensure correct order.
        """
        instance = snn.Instance(dt=self.dt)
        linear = snn.Synapse(10, 10, instance=instance)
        li = snn.ReadoutNeuron(10, instance=instance)

        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63

        spikes = torch.zeros(10, 110, 10)
        for idx in range(10):
            spikes[:, idx * 10 + 5, idx] = 1

        i_handle = linear(snn.NeuronHandle(spikes))
        v_handle = li(i_handle)

        self.assertTrue(v_handle.v_cadc is None)
        self.assertTrue(v_handle.v_madc is None)

        snn.run(instance, 110)

        # Assert types and shapes
        self.assertIsInstance(v_handle.v_cadc, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(v_handle.v_cadc.shape),
                torch.tensor([10, 110 + 1, 10])))
        self.assertTrue(v_handle.v_madc is None)

    def test_record_madc(self):
        """
        Test MADC recording.

        TODO:
            - Ensure correct neuron is recorded.
        """
        instance = snn.Instance(dt=self.dt)
        linear = snn.Synapse(10, 10, instance=instance)
        li = snn.ReadoutNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            instance=instance)

        spikes = torch.zeros(10, 110, 10)
        i_handle = linear(snn.NeuronHandle(spikes))
        li(i_handle)

        # TODO: Adjust as soon `to_dense` for MADC samples is implemented.
        with self.assertRaises(NotImplementedError):
            snn.run(instance, 110)

        # Only one module can record
        instance = snn.Instance(dt=self.dt)
        linear_1 = snn.Synapse(10, 10, instance=instance)
        li_1 = snn.ReadoutNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            instance=instance)
        linear_2 = snn.Synapse(10, 10, instance=instance)
        li_2 = snn.ReadoutNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            instance=instance)

        spikes = torch.zeros(10, 110, 10)
        i_handle_1 = linear_1(snn.NeuronHandle(spikes))
        v_handle_1 = li_1(i_handle_1)
        i_handle_2 = linear_2(v_handle_1)
        li_2(i_handle_2)

        # Execute
        with self.assertRaises(RuntimeError):  # Expect RuntimeError
            snn.run(instance, 110)

    def test_events_on_membrane(self):
        """
        Test whether events arrive at desired membrane.
        """
        pass


class TestSynapse(HWTestCase):
    """ Test hxtorch.snn.modules.Synapse """

    def test_output_type(self):
        """
        Test Synapse returns the expected handle
        """
        instance = snn.Instance()
        synapse = snn.Synapse(44, 33, instance)
        # Test output handle
        synapse_handle = synapse(snn.NeuronHandle(spikes=torch.zeros(10, 44)))
        self.assertTrue(isinstance(synapse_handle, snn.SynapseHandle))
        self.assertIsNone(synapse_handle.current)

    def test_weight_shape(self):
        """
        Test synapse weights are of correct shape.
        """
        instance = snn.Instance()
        synapse = snn.Synapse(44, 33, instance)
        # Test shape
        self.assertEqual(synapse.weight.shape[0], 33)
        self.assertEqual(synapse.weight.shape[1], 44)

    def test_weight_reset(self):
        """
        Test reset_parameters is working correctly
        """
        instance = snn.Instance()
        synapse = snn.Synapse(44, 33, instance)
        # Test weights are not zero (weight are initialized as zero and
        # reset_params is called implicitly)
        self.assertFalse(torch.equal(torch.zeros(33, 44), synapse.weight))

    def test_signed_projection(self):
        """
        Test synapse is represented on hardware as expected
        """
        pass


class TestBatchDropout(HWTestCase):
    """ Test hxtorch.snn.modules.BatchDropout """

    def test_output_type(self):
        pass


class TestInputNeuron(HWTestCase):
    """ Test hxtorch.snn.modules.InputNeuron """
    pass


if __name__ == "__main__":
    unittest.main()
