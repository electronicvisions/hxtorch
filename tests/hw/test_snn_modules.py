"""
Test snn.HXNeuron
"""
import unittest
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from dlens_vx_v3 import lola, halco
import hxtorch
from hxtorch import snn
from hxtorch.snn.utils import calib_helper

hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)


class TestHXModules(unittest.TestCase):
    """ Test HXModule """

    def test_forward(self):
        """
        Test Synapse returns the expected handle and registers module
        properly.
        """
        experiment = snn.Experiment(mock=True)
        module = snn.HXModule(experiment, None)
        # Test output handle
        input_handle = snn.TensorHandle()
        synapse_handle = module(input_handle)
        self.assertTrue(isinstance(synapse_handle, snn.TensorHandle))
        # Test module is registered in experiment
        self.assertTrue(module in experiment.modules.nodes)
        # Test handles are assigned properly
        sources = [
            e["handle"] for u, v, e in experiment.modules.graph.in_edges(
                experiment.modules.nodes[module], data=True)]
        targets = [
            e["handle"] for u, v, e in experiment.modules.graph.out_edges(
                experiment.modules.nodes[module], data=True)]
        self.assertEqual(sources, [input_handle])
        self.assertEqual(targets, [synapse_handle])

    def test_prepare_function(self):
        """
        Test prepare_function strips func properly
        """
        # Test with normal function with params and hw_results
        def func(input, params=None, hw_data=None):
            return input, params, hw_data

        experiment = snn.Experiment(mock=True)
        module = snn.HXModule(experiment, func)
        module.extra_kwargs.update({"params": "new_params"})
        new_func = module.func
        output, params, result_ret = new_func((None,), "hw_result")
        self.assertEqual(params, "new_params")
        self.assertEqual(result_ret, "hw_result")
        self.assertIsNone(output)

        # Test with normal function with params and without hw_results
        def func(input, params=None):
            return input, params

        experiment = snn.Experiment(mock=True)
        module = snn.HXModule(experiment, func)
        module.extra_kwargs.update({"params": "new_params"})
        new_func = module.func
        output, params_ret = new_func((None,))
        self.assertEqual(params_ret, "new_params")
        self.assertIsNone(output)

        # Test with normal function without params and with hw_results
        def func(input, hw_data=None):
            return input, hw_data

        experiment = snn.Experiment(mock=True)
        module = snn.HXModule(experiment, func)
        module.extra_kwargs.update({"params": "new_params"})
        new_func = module.func
        output, result_ret = new_func((None,), "hw_result")
        self.assertEqual(result_ret, "hw_result")
        self.assertIsNone(output)

        # Test with autograd function
        class Func(torch.autograd.Function):
            def forward(ctx, input, params=None, hw_data=None):
                return input, params, hw_data

            def backward(ctx, grad):
                return grad

        experiment = snn.Experiment(mock=True)
        module = snn.HXModule(experiment, Func)
        module.extra_kwargs.update({"params": "new_params"})
        new_func = module.func
        output, params, result_ret = new_func((None,), "hw_result")
        self.assertEqual(params, "new_params")
        self.assertEqual(result_ret, "hw_result")
        self.assertIsNone(output)

        # Test with autograd function, params and no hw_data
        class Func(torch.autograd.Function):
            def forward(ctx, input, params=None):
                return input, params

            def backward(ctx, grad):
                return grad

        experiment = snn.Experiment(mock=True)
        module = snn.HXModule(experiment, Func)
        module.extra_kwargs.update({"params": "new_params"})
        new_func = module.func
        output, params = new_func((None,), "hw_result")
        self.assertEqual(params, "new_params")
        self.assertIsNone(output)

        # Test with autograd function, no params and hw_data
        class Func(torch.autograd.Function):
            def forward(ctx, input, hw_data=None):
                return input, hw_data

            def backward(ctx, grad):
                return grad

        experiment = snn.Experiment(mock=True)
        module = snn.HXModule(experiment, Func)
        new_func = module.func
        output, hw_result = new_func((None,), "hw_result")
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

        experiment = snn.Experiment(mock=True)
        module = snn.HXModule(experiment, func)
        module.extra_kwargs.update({"params": "new_params"})
        module.extra_args = (1, 2)

        # Input and output handles
        input_handle = snn.NeuronHandle(torch.zeros(10, 5))
        output_handle = snn.NeuronHandle()

        # Execute
        module.exec_forward(
            input_handle, output_handle, {module.descriptor: "hw_result"})
        self.assertTrue(torch.equal(input_handle.spikes, output_handle.spikes))

        # Normal function
        def func(input, one, two, params=None, hw_data=None):
            self.assertEqual(params, "new_params")
            self.assertEqual(hw_data, "hw_result")
            self.assertEqual((one, two), (1, 2))
            return input

        experiment = snn.Experiment(mock=True)
        module = snn.HXModule(experiment, func)
        module.extra_kwargs.update({"params": "new_params"})
        module.extra_args = (1, 2)

        # Input and output handles
        input_handle = snn.NeuronHandle(torch.zeros(10, 5))
        output_handle = snn.NeuronHandle()

        # Execute
        module.exec_forward(
            input_handle, output_handle, {module.descriptor: "hw_result"})
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

        experiment = snn.Experiment(mock=True)
        module = snn.HXModule(experiment, Func)
        module.extra_kwargs.update({"params": "new_params"})
        module.extra_args = (1, 2)

        # Input and output handles
        input_handle = snn.NeuronHandle(torch.zeros(10, 5))
        output_handle = snn.NeuronHandle()

        # Execute
        module.exec_forward(
            input_handle, output_handle, {module.descriptor: "hw_result"})
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

        experiment = snn.Experiment(mock=False)
        module = snn.HXModule(experiment, Func)
        module.extra_kwargs.update({"params": "new_params"})
        module.extra_args = (1, 2)

        # Input and output handles
        input_handle = snn.NeuronHandle(torch.zeros(10, 5))
        output_handle = snn.NeuronHandle()

        # Execute
        module.exec_forward(
            input_handle, output_handle, {module.descriptor: "hw_result"})
        # Hw result should be assigned here
        self.assertTrue(output_handle.spikes, "hw_result")


class TestHXModuleWrapper(unittest.TestCase):
    """ TEst HXModuleWrapper """

    def test_contains(self):
        """ Test wrapper contains module """
        # Experiment
        experiment = snn.Experiment()

        # Modules
        linear = snn.Synapse(10, 10, experiment=experiment)
        lif = snn.Neuron(10, experiment=experiment)

        wrapper = snn.HXModuleWrapper(experiment, [linear, lif], None)

        # Should contain
        self.assertTrue(wrapper.contains([linear]))
        self.assertTrue(wrapper.contains([lif]))
        self.assertTrue(wrapper.contains([linear, lif]))

        # Should not contain
        lif2 = snn.Neuron(10, experiment=experiment)
        self.assertFalse(wrapper.contains([lif2]))
        self.assertFalse(wrapper.contains([linear, lif2]))

    def test_extra_args(self):
        """ test extra args """
        # Experiment
        experiment = snn.Experiment()

        # Modules
        linear1 = snn.Synapse(10, 10, experiment=experiment)
        lif1 = snn.Neuron(10, experiment=experiment)
        linear2 = snn.Synapse(10, 10, experiment=experiment)
        lif2 = snn.Neuron(10, experiment=experiment)

        wrapper = snn.HXModuleWrapper(
            experiment, [linear1, lif1, linear2, lif2], None)

        self.assertEqual(
            wrapper.extra_args, linear1.extra_args + linear2.extra_args)
        self.assertEqual(
            wrapper.extra_kwargs,
            {"params1": lif1.extra_kwargs["params"],
             "dt1": lif1.extra_kwargs["dt"],
             "params2": lif2.extra_kwargs["params"],
             "dt2": lif2.extra_kwargs["dt"]})

    def test_update(self):
        """ Test update modules """
        # Experiment
        experiment = snn.Experiment()

        # Modules
        linear1 = snn.Synapse(10, 10, experiment=experiment)
        lif1 = snn.Neuron(10, experiment=experiment)
        wrapper = snn.HXModuleWrapper(
            experiment, [linear1, lif1], None)
        self.assertEqual([linear1, lif1], wrapper.modules)

        linear2 = snn.Synapse(10, 10, experiment=experiment)
        lif2 = snn.Neuron(10, experiment=experiment)
        wrapper.update([linear2, lif2])
        self.assertEqual([linear2, lif2], wrapper.modules)

    def test_exec_forward(self):
        """ """
        in_tensor = torch.zeros(10, 5, 10)

        def func(input, arg1, arg2, arg3):
            self.assertTrue(torch.equal(input, in_tensor))
            self.assertEqual(arg1, "w1")
            self.assertEqual(arg2, "b1")
            self.assertEqual(arg3, "w2")
            return "syn1", ("z1", "v1"), "syn2", "nrn2"

        # Experiment
        experiment = snn.Experiment()

        # Modules
        linear1 = snn.Synapse(10, 10, experiment=experiment)
        lif1 = snn.Neuron(10, experiment=experiment)
        linear2 = snn.Synapse(10, 10, experiment=experiment)
        lif2 = snn.Neuron(10, experiment=experiment)
        # Change args before function assignment
        linear1.extra_args = ("w1", "b1")
        linear2.extra_args = ("w2",)

        wrapper = snn.HXModuleWrapper(
            experiment, [linear1, lif1, linear2, lif2], func)

        # Fake grenade descriptiors
        linear1.descriptor = "linear1"
        lif1.descriptor = "lif1"
        linear2.descriptor = "linear2"
        lif2.descriptor = "lif2"

        # Forward
        in_h = snn.NeuronHandle(in_tensor)
        syn1 = linear1(in_h)
        nrn1 = lif1(syn1)
        syn2 = linear2(nrn1)
        nrn2 = lif2(syn2)

        inputs = (in_h,)
        outputs = (syn1, nrn1, syn2, nrn2)

        # Execute forward
        wrapper.exec_forward(inputs, outputs, {})

        self.assertEqual(syn1.graded_spikes, "syn1")
        self.assertEqual(nrn1.spikes, "z1")
        self.assertEqual(nrn1.v_cadc, "v1")
        self.assertEqual(syn2.graded_spikes, "syn2")
        self.assertEqual(nrn2.spikes, "nrn2")

        # Test with HW data
        def func(input, arg1, arg2, arg3, hw_data):
            self.assertTrue(torch.equal(input, in_tensor))
            self.assertEqual(arg1, "w1")
            self.assertEqual(arg2, "b1")
            self.assertEqual(arg3, "w2")
            self.assertEqual(
                hw_data, (("syn1",), ("nrn1",), ("syn2",), ("nrn2",)))
            return "syn1", ("z1", "v1"), "syn2", "nrn2"

        # Experiment
        experiment = snn.Experiment()

        # Modules
        linear1 = snn.Synapse(10, 10, experiment=experiment)
        lif1 = snn.Neuron(10, experiment=experiment)
        linear2 = snn.Synapse(10, 10, experiment=experiment)
        lif2 = snn.Neuron(10, experiment=experiment)
        # Change args before function assignment
        linear1.extra_args = ("w1", "b1")
        linear2.extra_args = ("w2",)

        wrapper = snn.HXModuleWrapper(
            experiment, [linear1, lif1, linear2, lif2], func)

        # Fake greande descriptiors
        linear1.descriptor = "linear1"
        lif1.descriptor = "lif1"
        linear2.descriptor = "linear2"
        lif2.descriptor = "lif2"

        # Forward
        in_h = snn.NeuronHandle(in_tensor)
        syn1 = linear1(in_h)
        nrn1 = lif1(syn1)
        syn2 = linear2(nrn1)
        nrn2 = lif2(syn2)

        inputs = (in_h,)
        outputs = (syn1, nrn1, syn2, nrn2)
        hw_data = {
            "linear1": ("syn1",),
            "lif1": ("nrn1",),
            "linear2": ("syn2",),
            "lif2": ("nrn2",)}

        # Execute forward
        wrapper.exec_forward(inputs, outputs, hw_data)

        self.assertEqual(syn1.graded_spikes, "syn1")
        self.assertEqual(nrn1.spikes, "z1")
        self.assertEqual(nrn1.v_cadc, "v1")
        self.assertEqual(syn2.graded_spikes, "syn2")
        self.assertEqual(nrn2.spikes, "nrn2")


class HWTestCase(unittest.TestCase):
    """ HW setup """

    dt = 1.0e-6
    plot_path = Path(__file__).parent.joinpath("plots")

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()


class TestNeuron(HWTestCase):
    """ Test hxtorch.snn.modules.Neuron """

    def test_output_type(self):
        """
        Test neuron returns the expected handle
        """
        experiment = snn.Experiment(
            calib_path=calib_helper.nightly_calib_path())
        neuron = snn.Neuron(44, experiment)
        # Test output handle
        neuron_handle = neuron(snn.SynapseHandle(torch.zeros(10, 44)))
        self.assertTrue(isinstance(neuron_handle, snn.NeuronHandle))
        self.assertIsNone(neuron_handle.spikes)
        self.assertIsNone(neuron_handle.v_cadc)
        self.assertIsNone(neuron_handle.v_madc)

    def test_register_hw_entity(self):
        """
        Test hw entitiy is registered as expected
        """
        scales = torch.rand(10)
        offsets = torch.rand(10)

        experiment = snn.Experiment()
        neuron = snn.Neuron(10, experiment)
        neuron.register_hw_entity()
        self.assertEqual(experiment.id_counter, 10)
        self.assertEqual(len(experiment._populations), 1)
        self.assertEqual(experiment._populations[0], neuron)

        experiment = snn.Experiment()
        neuron = snn.Neuron(
            10, experiment, trace_offset=offsets, trace_scale=scales)
        neuron.register_hw_entity()
        self.assertTrue(torch.equal(neuron.scale, scales))
        self.assertTrue(torch.equal(neuron.offset, offsets))

        scales_dict = {}
        offsets_dict = {}
        for i in range(scales.shape[0]):
            scales_dict[halco.LogicalNeuronOnDLS(
                halco.LogicalNeuronCompartments(
                    {halco.CompartmentOnLogicalNeuron():
                        [halco.AtomicNeuronOnLogicalNeuron()]}
                ),
                halco.AtomicNeuronOnDLS(halco.EnumRanged_512_(i)))] \
                = scales[i]
            offsets_dict[halco.LogicalNeuronOnDLS(
                halco.LogicalNeuronCompartments(
                    {halco.CompartmentOnLogicalNeuron():
                        [halco.AtomicNeuronOnLogicalNeuron()]}
                ), halco.AtomicNeuronOnDLS(halco.EnumRanged_512_(i)))] \
                = offsets[i]
        experiment = snn.Experiment()
        neuron = snn.Neuron(
            10, experiment, trace_offset=offsets_dict, trace_scale=scales_dict)
        neuron.register_hw_entity()
        self.assertTrue(torch.equal(neuron.scale, scales))
        self.assertTrue(torch.equal(neuron.offset, offsets))

    def test_record_spikes(self):
        """
        Test spike recording with bypass mode.
        """
        # Enable bypass
        experiment = snn.Experiment(
            dt=self.dt,
            calib_path=calib_helper.nightly_calib_path())
        experiment._chip = lola.Chip.default_neuron_bypass

        # Modules
        linear = snn.Synapse(10, 10, experiment=experiment)
        lif = snn.Neuron(
            10, enable_cadc_recording=False,  experiment=experiment)

        # Weights
        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63

        # Inputs
        spikes = torch.zeros(110, 10, 10)
        for idx in range(10):
            spikes[idx * 10 + 5, :, idx] = 1

        # Forward
        i_handle = linear(snn.NeuronHandle(spikes))
        s_handle = lif(i_handle)

        self.assertTrue(s_handle.spikes is None)
        self.assertTrue(s_handle.v_cadc is None)
        self.assertTrue(s_handle.v_madc is None)

        # Execute
        snn.run(experiment, 110)

        # Assert types and shapes
        self.assertIsInstance(s_handle.spikes, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([110 + 1, 10, 10])))
        self.assertTrue(s_handle.v_cadc is None)
        self.assertTrue(s_handle.v_madc is None)

        # Assert data
        spike_times = torch.nonzero(s_handle.spikes)
        self.assertEqual(spike_times.shape[0], 10 * 10)

        i = 0
        for nrn in range(10):
            for b in range(10):
                self.assertEqual(b, spike_times[i, 1])
                self.assertEqual(5 + 10 * nrn, spike_times[i, 0])
                self.assertEqual(nrn, spike_times[i, 2])
                i += 1

    def test_record_cadc(self):
        """
        Test CADC recording.

        TODO:
            - Ensure correct order.
        """
        experiment = snn.Experiment(
            dt=self.dt,
            calib_path=calib_helper.nightly_calib_path())
        # Modules
        linear = snn.Synapse(10, 10, experiment=experiment)
        lif = snn.Neuron(10, experiment=experiment)
        # Weights
        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63
        # Inputs
        spikes = torch.zeros(110, 10, 10)
        for idx in range(10):
            spikes[idx * 10 + 5, :, idx] = 1

        # Forward
        i_handle = linear(snn.NeuronHandle(spikes))
        s_handle = lif(i_handle)

        self.assertTrue(s_handle.spikes is None)
        self.assertTrue(s_handle.v_cadc is None)
        self.assertTrue(s_handle.v_madc is None)

        # Execute
        snn.run(experiment, 110)

        # Assert types and shapes
        self.assertIsInstance(s_handle.spikes, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([110 + 1, 10, 10])))
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.v_cadc.shape),
                torch.tensor([110 + 1, 10, 10])))
        self.assertTrue(s_handle.v_madc is None)

    def test_record_madc(self):
        """
        Test MADC recording.

        TODO:
            - Ensure correct neuron is recorded.
        """
        experiment = snn.Experiment(
            dt=self.dt,
            calib_path=calib_helper.nightly_calib_path())
        linear = snn.Synapse(10, 10, experiment=experiment)
        lif = snn.Neuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)

        spikes = torch.zeros(110, 10, 10)
        i_handle = linear(snn.NeuronHandle(spikes))
        lif(i_handle)

        # TODO: Adjust as soon `to_dense` for MADC samples is implemented.
        with self.assertRaises(NotImplementedError):
            snn.run(experiment, 110)

        # Only one module can record
        experiment = snn.Experiment(
            dt=self.dt,
            calib_path=calib_helper.nightly_calib_path())
        linear_1 = snn.Synapse(10, 10, experiment=experiment)
        lif_1 = snn.Neuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        linear_2 = snn.Synapse(10, 10, experiment=experiment)
        lif_2 = snn.Neuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)

        spikes = torch.zeros(110, 10, 10)
        i_handle_1 = linear_1(snn.NeuronHandle(spikes))
        s_handle_1 = lif_1(i_handle_1)
        i_handle_2 = linear_2(s_handle_1)
        lif_2(i_handle_2)

        # Execute
        with self.assertRaises(RuntimeError):  # Expect RuntimeError
            snn.run(experiment, 110)

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
        experiment = snn.Experiment()
        neuron = snn.ReadoutNeuron(44, experiment)
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
        experiment = snn.Experiment(
            dt=self.dt,
            calib_path=calib_helper.nightly_calib_path())
        linear = snn.Synapse(10, 10, experiment=experiment)
        li = snn.ReadoutNeuron(10, experiment=experiment)

        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63

        spikes = torch.zeros(110, 10, 10)
        for idx in range(10):
            spikes[idx * 10 + 5, :, idx] = 1

        i_handle = linear(snn.NeuronHandle(spikes))
        v_handle = li(i_handle)

        self.assertTrue(v_handle.v_cadc is None)
        self.assertTrue(v_handle.v_madc is None)

        snn.run(experiment, 110)

        # Assert types and shapes
        self.assertIsInstance(v_handle.v_cadc, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(v_handle.v_cadc.shape),
                torch.tensor([110 + 1, 10, 10])))
        self.assertTrue(v_handle.v_madc is None)

    def test_record_madc(self):
        """
        Test MADC recording.

        TODO:
            - Ensure correct neuron is recorded.
        """
        experiment = snn.Experiment(
            dt=self.dt,
            calib_path=calib_helper.nightly_calib_path())
        linear = snn.Synapse(10, 10, experiment=experiment)
        li = snn.ReadoutNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)

        spikes = torch.zeros(110, 10, 10)
        i_handle = linear(snn.NeuronHandle(spikes))
        li(i_handle)

        # TODO: Adjust as soon `to_dense` for MADC samples is implemented.
        with self.assertRaises(NotImplementedError):
            snn.run(experiment, 110)

        # Only one module can record
        experiment = snn.Experiment(
            dt=self.dt,
            calib_path=calib_helper.nightly_calib_path())
        linear_1 = snn.Synapse(10, 10, experiment=experiment)
        li_1 = snn.ReadoutNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        linear_2 = snn.Synapse(10, 10, experiment=experiment)
        li_2 = snn.ReadoutNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)

        spikes = torch.zeros(110, 10, 10)
        i_handle_1 = linear_1(snn.NeuronHandle(spikes))
        v_handle_1 = li_1(i_handle_1)
        i_handle_2 = linear_2(v_handle_1)
        li_2(i_handle_2)

        # Execute
        with self.assertRaises(RuntimeError):  # Expect RuntimeError
            snn.run(experiment, 110)

    def test_events_on_membrane(self):
        """
        Test whether events arrive at desired membrane.
        """
        pass


class TestIAFNeuron(HWTestCase):
    """ Test hxtorch.snn.modules.IAFNeuron """

    def test_output_type(self):
        """
        Test neuron returns the expected handle
        """
        experiment = snn.Experiment()
        neuron = snn.IAFNeuron(44, experiment)
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
        experiment = snn.Experiment(
            dt=self.dt,
            calib_path=calib_helper.nightly_calib_path())
        experiment._chip = lola.Chip.default_neuron_bypass
        # Modules
        linear = snn.Synapse(10, 10, experiment=experiment)
        lif = snn.IAFNeuron(
            10, params=snn.functional.CUBAIAFParams(0./10e-6, 0./10e-6),
            enable_cadc_recording=False, experiment=experiment)
        # Weights
        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63
        # Inputs
        spikes = torch.zeros(110, 10, 10)
        for idx in range(10):
            spikes[idx * 10 + 5, :, idx] = 1
        # Forward
        i_handle = linear(snn.NeuronHandle(spikes))
        s_handle = lif(i_handle)

        self.assertTrue(s_handle.spikes is None)
        self.assertTrue(s_handle.v_cadc is None)
        self.assertTrue(s_handle.v_madc is None)

        # Execute
        snn.run(experiment, 110)

        # Assert types and shapes
        self.assertIsInstance(s_handle.spikes, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([110 + 1, 10, 10])))
        self.assertTrue(s_handle.v_cadc is None)
        self.assertTrue(s_handle.v_madc is None)

        # Assert data
        spike_times = torch.nonzero(s_handle.spikes)
        self.assertEqual(spike_times.shape[0], 10 * 10)

        i = 0
        for nrn in range(10):
            for b in range(10):
                self.assertEqual(b, spike_times[i, 1])
                self.assertEqual(5 + 10 * nrn, spike_times[i, 0])
                self.assertEqual(nrn, spike_times[i, 2])
                i += 1

    def test_record_cadc(self):
        """
        Test CADC recording.
        TODO:
            - Ensure correct order.
        """
        experiment = snn.Experiment(
            dt=self.dt,
            calib_path=calib_helper.nightly_calib_path())
        # Modules
        linear = snn.Synapse(10, 10, experiment=experiment)
        lif = snn.IAFNeuron(
            10, enable_cadc_recording=True, experiment=experiment)
        # Weights
        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 50
        # Inputs
        spikes = torch.zeros(110, 10, 10)
        for idx in range(10):
            spikes[idx * 10 + 5, :, idx] = 1
        # Forward
        i_handle = linear(snn.NeuronHandle(spikes))
        s_handle = lif(i_handle)

        self.assertTrue(s_handle.spikes is None)
        self.assertTrue(s_handle.v_cadc is None)
        self.assertTrue(s_handle.v_madc is None)

        # Execute
        snn.run(experiment, 110)
        # Assert types and shapes
        self.assertIsInstance(s_handle.spikes, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([110 + 1, 10, 10])))
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.v_cadc.shape),
                torch.tensor([110 + 1, 10, 10])))
        self.assertTrue(s_handle.v_madc is None)

        # plot
        self.plot_path.mkdir(exist_ok=True)
        trace = s_handle.v_cadc[:, 0].detach().numpy()
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., trace.shape[0]), trace)
        plt.savefig(self.plot_path.joinpath("./cuba_iaf_dynamics.png"))

    def test_record_madc(self):
        """
        Test MADC recording.

        TODO:
            - Ensure correct neuron is recorded.
        """
        experiment = snn.Experiment(
            dt=self.dt,
            calib_path=calib_helper.nightly_calib_path())
        linear = snn.Synapse(10, 10, experiment=experiment)
        lif = snn.IAFNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        spikes = torch.zeros(110, 10, 10)
        i_handle = linear(snn.NeuronHandle(spikes))
        lif(i_handle)
        # TODO: Adjust as soon `to_dense` for MADC samples is implemented.
        with self.assertRaises(NotImplementedError):
            snn.run(experiment, 110)
        # Only one module can record
        experiment = snn.Experiment(
            dt=self.dt,
            calib_path=calib_helper.nightly_calib_path())
        linear_1 = snn.Synapse(10, 10, experiment=experiment)
        lif_1 = snn.IAFNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        linear_2 = snn.Synapse(10, 10, experiment=experiment)
        lif_2 = snn.IAFNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        spikes = torch.zeros(110, 10, 10)
        i_handle_1 = linear_1(snn.NeuronHandle(spikes))
        s_handle_1 = lif_1(i_handle_1)
        i_handle_2 = linear_2(s_handle_1)
        lif_2(i_handle_2)
        # Execute
        with self.assertRaises(RuntimeError):  # Expect RuntimeError
            snn.run(experiment, 110)

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


class TestSynapse(HWTestCase):
    """ Test hxtorch.snn.modules.Synapse """

    def test_output_type(self):
        """
        Test Synapse returns the expected handle
        """
        experiment = snn.Experiment()
        synapse = snn.Synapse(44, 33, experiment)
        # Test output handle
        synapse_handle = synapse(snn.NeuronHandle(spikes=torch.zeros(10, 44)))
        self.assertTrue(isinstance(synapse_handle, snn.SynapseHandle))
        self.assertIsNone(synapse_handle.graded_spikes)

    def test_weight_shape(self):
        """
        Test synapse weights are of correct shape.
        """
        experiment = snn.Experiment()
        synapse = snn.Synapse(44, 33, experiment)
        # Test shape
        self.assertEqual(synapse.weight.shape[0], 33)
        self.assertEqual(synapse.weight.shape[1], 44)

    def test_weight_reset(self):
        """
        Test reset_parameters is working correctly
        """
        experiment = snn.Experiment()
        synapse = snn.Synapse(44, 33, experiment)
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
        """ Test BatchDropout returns expected handle """
        experiment = snn.Experiment()
        dropout = snn.BatchDropout(33, 0.5, experiment)
        # Test output handle
        dropout_handle = dropout(snn.NeuronHandle(spikes=torch.zeros(10, 44)))
        self.assertTrue(isinstance(dropout_handle, snn.NeuronHandle))
        self.assertIsNone(dropout_handle.spikes)
        self.assertIsNone(dropout_handle.current)
        self.assertIsNone(dropout_handle.v_cadc)
        self.assertIsNone(dropout_handle.v_madc)

    def test_set_mask(self):
        """ Test mask is updated properly """
        experiment = snn.Experiment()
        dropout = snn.BatchDropout(33, 0.5, experiment)

        # train mode
        dropout.train()
        mask1 = dropout.set_mask()
        self.assertTrue(torch.equal(mask1, dropout._mask))
        mask2 = dropout.set_mask()
        self.assertFalse(torch.equal(mask1, mask2))

        # eval mode
        dropout.eval()
        mask1 = dropout.set_mask()
        mask2 = dropout.set_mask()
        self.assertTrue(torch.equal(mask1, mask2))
        self.assertTrue(
            torch.equal(mask1, torch.ones_like(mask1)))

        # Test correct mask is passed to func
        def bd(x, mask):
            return x, mask
        experiment = snn.Experiment()
        dropout = snn.BatchDropout(33, 0.5, experiment, bd)
        dropout.set_mask()
        input = torch.zeros(10, 10, 33)
        output, mask1 = dropout.func((input,))
        self.assertTrue(torch.equal(mask1, dropout._mask))
        self.assertTrue(torch.equal(output, input))
        new_mask = dropout.set_mask()
        input, mask2 = dropout.func((input,))
        self.assertTrue(torch.equal(mask2, dropout._mask))
        self.assertTrue(torch.equal(new_mask, mask2))


class TestInputNeuron(HWTestCase):
    """ Test hxtorch.snn.modules.InputNeuron """
    pass


if __name__ == "__main__":
    unittest.main()
