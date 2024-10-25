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
from hxtorch import spiking as hxsnn
from hxtorch.spiking.utils import calib_helper
from hxtorch.spiking.execution_instance import ExecutionInstance

hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)
logger = hxtorch.logger.get("hxtorch.test.hw.test_spiking_modules")


class TestHXModules(unittest.TestCase):
    """ Test HXModule """

    def test_forward(self):
        """
        Test Synapse returns the expected handle and registers module
        properly.
        """
        experiment = hxsnn.Experiment(mock=True)
        module = hxsnn.HXModule(experiment, None)
        # Test output handle
        input_handle = hxsnn.TensorHandle()
        synapse_handle = module(input_handle)
        self.assertTrue(isinstance(synapse_handle, hxsnn.TensorHandle))
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

    def test_print(self):
        """ Test module printing """
        def func(input, params=None, hw_data=None):
            return input, params, hw_data
        experiment = hxsnn.Experiment(mock=True)
        module = hxsnn.HXModule(experiment, func)
        logger.INFO(module)

    def test_prepare_function(self):
        """
        Test prepare_function strips func properly
        """
        # Test with hw_results
        class HXModule(hxsnn.HXModule):
            def forward_func(self, input, hw_data=None):
                return input, hw_data

        experiment = hxsnn.Experiment(mock=True)
        module = HXModule(experiment)
        new_func = module.func
        output, result_ret = new_func((None,), "hw_result")
        self.assertEqual(result_ret, "hw_result")
        self.assertIsNone(output)

        # Test with without hw_results
        class HXModule(hxsnn.HXModule):
            def forward_func(selfw, input):
                return input

        experiment = hxsnn.Experiment(mock=True)
        module = HXModule(experiment)
        new_func = module.func
        output = new_func((None,), "hw_result")
        self.assertIsNone(output)

    def test_exec_forward(self):
        """
        Test execute_forward work as expected.
        """
        # Normal function
        class Module(hxsnn.HXModule):
            def forward_func(selfw, input, hw_data=None):
                self.assertEqual(selfw.param, "param1")
                self.assertEqual(hw_data, "hw_result")
                return input

        experiment = hxsnn.Experiment(mock=True)
        module = Module(experiment)
        module.param = "param1"

        # Input and output handles
        input_handle = hxsnn.NeuronHandle(torch.zeros(10, 5))
        output_handle = hxsnn.NeuronHandle()

        # Execute
        module.exec_forward(
            input_handle, output_handle, {module.descriptor: "hw_result"})
        self.assertTrue(torch.equal(input_handle.spikes, output_handle.spikes))


class TestHXModuleWrapper(unittest.TestCase):
    """ TEst HXModuleWrapper """

    def test_contains(self):
        """ Test wrapper contains module """
        # Experiment
        experiment = hxsnn.Experiment()

        # Modules
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        lif = hxsnn.Neuron(10, experiment=experiment)

        wrapper = hxsnn.HXModuleWrapper(experiment, linear=linear, lif=lif)

        # Should contain
        self.assertTrue(wrapper.contains(linear))
        self.assertTrue(wrapper.contains(lif))
        self.assertTrue(wrapper.contains([linear, lif]))

        # Should not contain
        lif2 = hxsnn.Neuron(10, experiment=experiment)
        self.assertFalse(wrapper.contains(lif2))
        self.assertFalse(wrapper.contains([lif2]))
        self.assertFalse(wrapper.contains([linear, lif2]))

    def test_print(self):
        """ Test module printing """
        experiment = hxsnn.Experiment()
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        lif = hxsnn.Neuron(10, experiment=experiment)
        module = hxsnn.HXModuleWrapper(experiment, linear=linear, lif=lif)
        logger.INFO(module)

    def test_update(self):
        """ Test update modules """
        # Experiment
        experiment = hxsnn.Experiment()

        # Modules
        linear1 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif1 = hxsnn.Neuron(10, experiment=experiment)
        wrapper = hxsnn.HXModuleWrapper(experiment, linear1=linear1, lif1=lif1)
        self.assertEqual({"linear1": linear1, "lif1": lif1}, wrapper.modules)

        linear2 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif2 = hxsnn.Neuron(10, experiment=experiment)
        wrapper.update(linear1=linear2, lif1=lif2)
        self.assertEqual({"linear1": linear2, "lif1": lif2}, wrapper.modules)

    def test_exec_forward(self):
        """ """
        in_tensor = torch.zeros(10, 5, 10)

        class Wrapper(hxsnn.HXModuleWrapper):
            def forward_func(selfw, input, hw_data=None):
                return (
                    hxsnn.SynapseHandle("syn1"), hxsnn.NeuronHandle("z1", "v1"),
                    hxsnn.SynapseHandle("syn2"), hxsnn.NeuronHandle("nrn2"))

        # Experiment
        experiment = hxsnn.Experiment()

        # Modules
        linear1 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif1 = hxsnn.Neuron(10, experiment=experiment)
        linear2 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif2 = hxsnn.Neuron(10, experiment=experiment)

        wrapper = Wrapper(
            experiment, linear1=linear1, lif1=lif1, linear2=linear2, lif2=lif2)

        # Fake grenade descriptors
        linear1.descriptor = "linear1"
        lif1.descriptor = "lif1"
        linear2.descriptor = "linear2"
        lif2.descriptor = "lif2"

        # Forward
        in_h = hxsnn.NeuronHandle(in_tensor)
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
        self.assertEqual(nrn1.membrane_cadc, "v1")
        self.assertEqual(syn2.graded_spikes, "syn2")
        self.assertEqual(nrn2.spikes, "nrn2")

        # Test with HW data
        class HWDataWrapper(hxsnn.HXModuleWrapper):
            def forward_func(selfw, input, hw_data=None):
                self.assertEqual(
                    hw_data, (("syn1",), ("nrn1",), ("syn2",), ("nrn2",)))
                return (
                    hxsnn.SynapseHandle("syn1"), hxsnn.NeuronHandle("z1", "v1"),
                    hxsnn.SynapseHandle("syn2"), hxsnn.NeuronHandle("nrn2"))

        # Experiment
        experiment = hxsnn.Experiment()

        # Modules
        linear1 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif1 = hxsnn.Neuron(10, experiment=experiment)
        linear2 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif2 = hxsnn.Neuron(10, experiment=experiment)

        wrapper = HWDataWrapper(
            experiment, linear1=linear1, lif1=lif1, linear2=linear2, lif2=lif2)

        # Fake grenade descriptors
        linear1.descriptor = "linear1"
        lif1.descriptor = "lif1"
        linear2.descriptor = "linear2"
        lif2.descriptor = "lif2"

        # Forward
        in_h = hxsnn.NeuronHandle(in_tensor)
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
        self.assertEqual(nrn1.membrane_cadc, "v1")
        self.assertEqual(syn2.graded_spikes, "syn2")
        self.assertEqual(nrn2.spikes, "nrn2")

    def test_forward(self):
        in_tensor = torch.zeros(10, 5, 10)

        # Test with HW data
        class Wrapper(hxsnn.HXModuleWrapper):
            def forward_func(selfw, input, hw_data=None):
                self.assertEqual((("syn",), ("nrn",)))
                return (
                    hxsnn.SynapseHandle("syn"), hxsnn.NeuronHandle("z1", "v1"))

        # Experiment
        experiment = hxsnn.Experiment()

        # Modules
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        lif = hxsnn.Neuron(10, experiment=experiment)
        wrapper = Wrapper(experiment, linear=linear, lif=lif)

        # forward
        inputs = hxsnn.NeuronHandle()
        syn = linear(inputs)
        nrn = lif(syn)
        wrapper()

        self.assertEqual(len(experiment.modules.wrappers), 1)
        self.assertEqual(experiment.modules.wrappers, {wrapper: "w_0"})

        # forward again
        inputs = hxsnn.NeuronHandle(in_tensor)
        syn = linear(inputs)
        nrn = lif(syn)
        wrapper()

        self.assertEqual(len(experiment.modules.wrappers), 1)
        self.assertEqual(experiment.modules.wrappers, {wrapper: "w_0"})


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
    """ Test hxtorch.hxsnn.modules.Neuron """

    def test_output_type(self):
        """
        Test neuron returns the expected handle
        """
        experiment = hxsnn.Experiment()
        neuron = hxsnn.Neuron(44, experiment)
        # Test output handle
        neuron_handle = neuron(hxsnn.SynapseHandle(torch.zeros(10, 44)))
        self.assertTrue(isinstance(neuron_handle, hxsnn.NeuronHandle))
        self.assertIsNone(neuron_handle.spikes)
        self.assertIsNone(neuron_handle.membrane_cadc)
        self.assertIsNone(neuron_handle.membrane_madc)

    def test_print(self):
        """ Test module printing """
        experiment = hxsnn.Experiment()
        module = hxsnn.Neuron(10, experiment=experiment)
        logger.INFO(module)

    def test_register_hw_entity(self):
        """
        Test hw entitiy is registered as expected
        """
        scales = torch.rand(10)
        offsets = torch.rand(10)

        experiment = hxsnn.Experiment()
        neuron = hxsnn.Neuron(10, experiment)
        neuron.register_hw_entity()
        self.assertEqual(
            experiment.default_execution_instance.id_counter, 10)
        self.assertEqual(len(experiment._populations), 1)
        self.assertEqual(experiment._populations[0], neuron)

        experiment = hxsnn.Experiment()
        neuron = hxsnn.Neuron(
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
        experiment = hxsnn.Experiment()
        neuron = hxsnn.Neuron(
            10, experiment, trace_offset=offsets_dict, trace_scale=scales_dict)
        neuron.register_hw_entity()
        self.assertTrue(torch.equal(neuron.scale, scales))
        self.assertTrue(torch.equal(neuron.offset, offsets))

    def test_record_spikes(self):
        """
        Test spike recording with bypass mode.
        """
        # Enable bypass
        experiment = hxsnn.Experiment(dt=self.dt)
        execution_instance = ExecutionInstance(
            # Hack that chip will not be overwritten
            calib_path=calib_helper.nightly_calib_path())
        execution_instance.chip = lola.Chip.default_neuron_bypass
        experiment.default_execution_instance = execution_instance

        # Modules
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        lif = hxsnn.Neuron(
            10, enable_cadc_recording=True,  experiment=experiment)

        # Weights
        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63

        # Inputs
        spikes = torch.zeros(110, 10, 10)
        for idx in range(10):
            spikes[idx * 10 + 5, :, idx] = 1

        # Forward
        i_handle = linear(hxsnn.NeuronHandle(spikes))
        s_handle = lif(i_handle)

        self.assertTrue(s_handle.spikes is None)
        self.assertTrue(s_handle.membrane_cadc is None)
        self.assertTrue(s_handle.membrane_madc is None)

        # Execute
        hxsnn.run(experiment, 110)

        # Assert types and shapes
        self.assertIsInstance(s_handle.spikes, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(s_handle.membrane_cadc is not None)
        self.assertTrue(s_handle.membrane_madc is None)

        # Assert data
        spike_times = torch.nonzero(s_handle.spikes)
        self.assertEqual(spike_times.shape[0], 10 * 10)

        i = 0
        for nrn in range(10):
            for b in range(10):
                self.assertEqual(b, spike_times[i, 1])
                # EA 2024-02-28: Sometimes spikes of first batch-entry seem to
                #                be delayed
                self.assertAlmostEqual(
                    5 + 10 * nrn, int(spike_times[i, 0]), delta=2)
                self.assertEqual(nrn, spike_times[i, 2])
                i += 1

    def test_record_cadc(self):
        """
        Test CADC recording.

        TODO:
            - Ensure correct order.
        """
        experiment = hxsnn.Experiment(dt=self.dt)
        # Modules
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        lif = hxsnn.Neuron(10, experiment=experiment)
        # Weights
        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63
        # Inputs
        spikes = torch.zeros(110, 10, 10)
        for idx in range(10):
            spikes[idx * 10 + 5, :, idx] = 1

        # Forward
        i_handle = linear(hxsnn.NeuronHandle(spikes))
        s_handle = lif(i_handle)

        self.assertTrue(s_handle.spikes is None)
        self.assertTrue(s_handle.membrane_cadc is None)
        self.assertTrue(s_handle.membrane_madc is None)

        # Execute
        hxsnn.run(experiment, 110)

        # Assert types and shapes
        self.assertIsInstance(s_handle.spikes, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.membrane_cadc.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(s_handle.membrane_madc is None)

    def test_record_madc(self):
        """
        Test MADC recording.

        TODO:
            - Ensure correct neuron is recorded.
        """
        experiment = hxsnn.Experiment(dt=self.dt)
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        lif = hxsnn.IAFNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        spikes = torch.zeros(110, 10, 10)
        i_handle = linear(hxsnn.NeuronHandle(spikes))
        s_handle = lif(i_handle)

        self.assertTrue(s_handle.spikes is None)
        self.assertTrue(s_handle.membrane_cadc is None)
        self.assertTrue(s_handle.membrane_madc is None)

        hxsnn.run(experiment, 110)

        # Assert types and shapes
        self.assertIsInstance(s_handle.spikes, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.membrane_cadc.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.membrane_madc.shape),
                torch.tensor([2, 3235, 10])))
        # Only one module can record
        experiment = hxsnn.Experiment(dt=self.dt)
        linear_1 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif_1 = hxsnn.Neuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        linear_2 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif_2 = hxsnn.Neuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)

        spikes = torch.zeros(110, 10, 10)
        i_handle_1 = linear_1(hxsnn.NeuronHandle(spikes))
        s_handle_1 = lif_1(i_handle_1)
        i_handle_2 = linear_2(s_handle_1)
        lif_2(i_handle_2)

        # Execute
        with self.assertRaises(RuntimeError):  # Expect RuntimeError
            hxsnn.run(experiment, 110)

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
    """ Test hxtorch.spiking.modules.ReadoutNeuron """

    def test_output_type(self):
        """
        Test ReadoutNeuron returns the expected handle
        """
        experiment = hxsnn.Experiment()
        neuron = hxsnn.ReadoutNeuron(44, experiment)
        # Test output handle
        neuron_handle = neuron(hxsnn.SynapseHandle(torch.zeros(10, 44)))
        self.assertTrue(isinstance(neuron_handle, hxsnn.ReadoutNeuronHandle))
        self.assertIsNone(neuron_handle.membrane_cadc)
        self.assertIsNone(neuron_handle.membrane_madc)

    def test_print(self):
        """ Test module printing """
        experiment = hxsnn.Experiment()
        module = hxsnn.ReadoutNeuron(10, experiment=experiment)
        logger.INFO(module)

    def test_record_cadc(self):
        """
        Test CADC recording.

        TODO:
            - Ensure correct order.
        """
        experiment = hxsnn.Experiment(dt=self.dt)
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        li = hxsnn.ReadoutNeuron(10, experiment=experiment)

        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63

        spikes = torch.zeros(110, 10, 10)
        for idx in range(10):
            spikes[idx * 10 + 5, :, idx] = 1

        i_handle = linear(hxsnn.NeuronHandle(spikes))
        v_handle = li(i_handle)

        self.assertTrue(v_handle.membrane_cadc is None)
        self.assertTrue(v_handle.membrane_madc is None)

        hxsnn.run(experiment, 110)

        # Assert types and shapes
        self.assertIsInstance(v_handle.membrane_cadc, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(v_handle.membrane_cadc.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(v_handle.membrane_madc is None)

    def test_record_madc(self):
        """
        Test MADC recording.

        TODO:
            - Ensure correct neuron is recorded.
        """
        experiment = hxsnn.Experiment(dt=self.dt)
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        li = hxsnn.ReadoutNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)

        spikes = torch.zeros(110, 10, 10)
        i_handle = linear(hxsnn.NeuronHandle(spikes))
        y_handle = li(i_handle)

        self.assertTrue(y_handle.membrane_cadc is None)
        self.assertTrue(y_handle.membrane_madc is None)

        hxsnn.run(experiment, 110)

        # Assert types and shapes
        self.assertTrue(
            torch.equal(
                torch.tensor(y_handle.membrane_cadc.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(
            torch.equal(
                torch.tensor(y_handle.membrane_madc.shape),
                torch.tensor([2, 3235, 10])))

        # Only one module can record
        experiment = hxsnn.Experiment(dt=self.dt)
        linear_1 = hxsnn.Synapse(10, 10, experiment=experiment)
        li_1 = hxsnn.ReadoutNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        linear_2 = hxsnn.Synapse(10, 10, experiment=experiment)
        li_2 = hxsnn.ReadoutNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)

        spikes = torch.zeros(110, 10, 10)
        i_handle_1 = linear_1(hxsnn.NeuronHandle(spikes))
        v_handle_1 = li_1(i_handle_1)
        i_handle_2 = linear_2(v_handle_1)
        li_2(i_handle_2)

        # Execute
        with self.assertRaises(RuntimeError):  # Expect RuntimeError
            hxsnn.run(experiment, 110)

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
        experiment = hxsnn.Experiment()
        neuron = hxsnn.IAFNeuron(44, experiment)
        # Test output handle
        neuron_handle = neuron(hxsnn.SynapseHandle(torch.zeros(10, 44)))
        self.assertTrue(isinstance(neuron_handle, hxsnn.NeuronHandle))
        self.assertIsNone(neuron_handle.spikes)
        self.assertIsNone(neuron_handle.membrane_cadc)
        self.assertIsNone(neuron_handle.membrane_madc)

    def test_print(self):
        """ Test module printing """
        experiment = hxsnn.Experiment()
        module = hxsnn.IAFNeuron(10, experiment=experiment)
        logger.INFO(module)

    def test_record_spikes(self):
        """
        Test spike recording with bypass mode.
        """
        # Enable bypass
        experiment = hxsnn.Experiment(dt=self.dt)
        execution_instance = ExecutionInstance(
            # Hack that chip will not be overwritten
            calib_path=calib_helper.nightly_calib_path())
        execution_instance.chip = lola.Chip.default_neuron_bypass
        experiment.default_execution_instance = execution_instance
        # Modules
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        lif = hxsnn.IAFNeuron(
            10, enable_cadc_recording=True, experiment=experiment)
        # Weights
        linear.weight.data.fill_(0.)
        for idx in range(10):
            linear.weight.data[idx, idx] = 63
        # Inputs
        spikes = torch.zeros(110, 10, 10)
        for idx in range(10):
            spikes[idx * 10 + 5, :, idx] = 1
        # Forward
        i_handle = linear(hxsnn.NeuronHandle(spikes))
        s_handle = lif(i_handle)

        self.assertTrue(s_handle.spikes is None)
        self.assertTrue(s_handle.membrane_cadc is None)
        self.assertTrue(s_handle.membrane_madc is None)

        # Execute
        hxsnn.run(experiment, 110)

        # Assert types and shapes
        self.assertIsInstance(s_handle.spikes, torch.Tensor)
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(s_handle.membrane_cadc is not None)
        self.assertTrue(s_handle.membrane_madc is None)

        # Assert data
        spike_times = torch.nonzero(s_handle.spikes)
        self.assertEqual(spike_times.shape[0], 10 * 10)

        i = 0
        for nrn in range(10):
            for b in range(10):
                self.assertEqual(b, spike_times[i, 1])
                # EA 2024-02-28: Sometimes spikes of first batch-entry seem to
                #                be delayed
                self.assertAlmostEqual(
                    5 + 10 * nrn, int(spike_times[i, 0]), delta=2)
                self.assertEqual(nrn, spike_times[i, 2])
                i += 1

    def test_record_cadc(self):
        """
        Test CADC recording.
        TODO:
            - Ensure correct order.
        """
        for use_dram in [False, True]:
            experiment = hxsnn.Experiment(dt=self.dt)
            # Modules
            linear = hxsnn.Synapse(10, 10, experiment=experiment)
            lif = hxsnn.IAFNeuron(
                10, enable_cadc_recording=True,
                enable_cadc_recording_placement_in_dram=use_dram,
                experiment=experiment)
            # Weights
            linear.weight.data.fill_(0.)
            for idx in range(10):
                linear.weight.data[idx, idx] = 50
            # Inputs
            spikes = torch.zeros(110, 10, 10)
            for idx in range(10):
                spikes[idx * 10 + 5, :, idx] = 1
            # Forward
            i_handle = linear(hxsnn.NeuronHandle(spikes))
            s_handle = lif(i_handle)

            self.assertTrue(s_handle.spikes is None)
            self.assertTrue(s_handle.membrane_cadc is None)
            self.assertTrue(s_handle.membrane_madc is None)

            # Execute
            hxsnn.run(experiment, 110)
            # Assert types and shapes
            self.assertIsInstance(s_handle.spikes, torch.Tensor)
            self.assertTrue(
                torch.equal(
                    torch.tensor(s_handle.spikes.shape),
                    torch.tensor([110, 10, 10])))
            self.assertTrue(
                torch.equal(
                    torch.tensor(s_handle.membrane_cadc.shape),
                    torch.tensor([110, 10, 10])))
            self.assertTrue(s_handle.membrane_madc is None)

            # plot
            self.plot_path.mkdir(exist_ok=True)
            trace = s_handle.membrane_cadc[:, 0].detach().numpy()
            fig, ax = plt.subplots()
            ax.plot(
                np.arange(0., trace.shape[0]), trace)
            plt.savefig(self.plot_path.joinpath(
                f"./cuba_iaf_dynamics_{int(use_dram)}.png"))

    def test_record_madc(self):
        """
        Test MADC recording.

        TODO:
            - Ensure correct neuron is recorded.
        """
        experiment = hxsnn.Experiment(dt=self.dt)
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        lif = hxsnn.IAFNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        spikes = torch.zeros(110, 10, 10)
        i_handle = linear(hxsnn.NeuronHandle(spikes))
        s_handle = lif(i_handle)

        self.assertTrue(s_handle.spikes is None)
        self.assertTrue(s_handle.membrane_cadc is None)
        self.assertTrue(s_handle.membrane_madc is None)

        hxsnn.run(experiment, 110)

        # Assert types and shapes
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.spikes.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.membrane_cadc.shape),
                torch.tensor([110, 10, 10])))
        self.assertTrue(
            torch.equal(
                torch.tensor(s_handle.membrane_madc.shape),
                torch.tensor([2, 3235, 10])))

        # Only one module can record
        experiment = hxsnn.Experiment(dt=self.dt)
        linear_1 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif_1 = hxsnn.IAFNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        linear_2 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif_2 = hxsnn.IAFNeuron(
            10, enable_madc_recording=True, record_neuron_id=1,
            experiment=experiment)
        spikes = torch.zeros(110, 10, 10)
        i_handle_1 = linear_1(hxsnn.NeuronHandle(spikes))
        s_handle_1 = lif_1(i_handle_1)
        i_handle_2 = linear_2(s_handle_1)
        lif_2(i_handle_2)
        # Execute
        with self.assertRaises(RuntimeError):  # Expect RuntimeError
            hxsnn.run(experiment, 110)

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
        experiment = hxsnn.Experiment()
        synapse = hxsnn.Synapse(44, 33, experiment)
        # Test output handle
        synapse_handle = synapse(hxsnn.NeuronHandle(spikes=torch.zeros(10, 44)))
        self.assertTrue(isinstance(synapse_handle, hxsnn.SynapseHandle))
        self.assertIsNone(synapse_handle.graded_spikes)

    def test_print(self):
        """ Test module printing """
        experiment = hxsnn.Experiment()
        module = hxsnn.Synapse(10, 22, experiment=experiment)
        logger.INFO(module)

    def test_weight_shape(self):
        """
        Test synapse weights are of correct shape.
        """
        experiment = hxsnn.Experiment()
        synapse = hxsnn.Synapse(44, 33, experiment)
        # Test shape
        self.assertEqual(synapse.weight.shape[0], 33)
        self.assertEqual(synapse.weight.shape[1], 44)

    def test_weight_reset(self):
        """
        Test reset_parameters is working correctly
        """
        experiment = hxsnn.Experiment()
        synapse = hxsnn.Synapse(44, 33, experiment)
        # Test weights are not zero (weight are initialized as zero and
        # reset_params is called implicitly)
        self.assertFalse(torch.equal(torch.zeros(33, 44), synapse.weight))

    def test_signed_projection(self):
        """
        Test synapse is represented on hardware as expected
        """
        pass


class TestSparseSynapse(HWTestCase):
    """ Test hxtorch.snn.modules.SparseSynapse """

    def test_output_type(self):
        """
        Test Synapse returns the expected handle
        """
        connections = (torch.randn(30, 44) < 0.1).float()
        experiment = hxsnn.Experiment()
        synapse = hxsnn.SparseSynapse(connections.to_sparse(), experiment)
        # Test output handle
        synapse_handle = synapse(
            hxsnn.NeuronHandle(spikes=torch.zeros(10, 44)))
        self.assertTrue(isinstance(synapse_handle, hxsnn.SynapseHandle))
        self.assertIsNone(synapse_handle.graded_spikes)

    def test_weight_shape(self):
        """
        Test synapse weights are of correct shape.
        """
        connections = (torch.randn(30, 44) < 0.1).float()
        experiment = hxsnn.Experiment()
        synapse = hxsnn.SparseSynapse(connections, experiment)
        # Test shape
        self.assertEqual(synapse.weight.shape[0], 44)
        self.assertEqual(synapse.weight.shape[1], 30)

    def test_weight_reset(self):
        """
        Test reset_parameters is working correctly
        """
        connections = (torch.randn(30, 44) < 0.1).float()
        experiment = hxsnn.Experiment()
        synapse = hxsnn.SparseSynapse(connections, experiment)
        # Test weights are not zero (weight are initialized as zero and
        # reset_params is called implicitly)
        self.assertFalse(
            torch.equal(torch.zeros(44, 30), synapse.weight.to_dense()))

    def test_execution(self):
        """
        Test synapse is represented on hardware as expected
        """
        connections = (torch.randn(30, 44) < 0.1).float()
        experiment = hxsnn.Experiment(dt=self.dt)
        linear = hxsnn.SparseSynapse(
            connections.to_sparse(), experiment=experiment)
        lif = hxsnn.ReadoutNeuron(44, experiment=experiment)
        spikes = torch.zeros(50, 1, 30)
        i_handle = linear(hxsnn.NeuronHandle(spikes))
        lif(i_handle)
        hxsnn.run(experiment, 110)


class TestBatchDropout(HWTestCase):
    """ Test hxtorch.snn.modules.BatchDropout """

    def test_output_type(self):
        """ Test BatchDropout returns expected handle """
        experiment = hxsnn.Experiment()
        dropout = hxsnn.BatchDropout(33, 0.5, experiment)
        # Test output handle
        dropout_handle = dropout(hxsnn.NeuronHandle(spikes=torch.zeros(10, 44)))
        self.assertTrue(isinstance(dropout_handle, hxsnn.NeuronHandle))
        self.assertIsNone(dropout_handle.spikes)
        self.assertIsNone(dropout_handle.current)
        self.assertIsNone(dropout_handle.membrane_cadc)
        self.assertIsNone(dropout_handle.membrane_madc)

    def test_print(self):
        """ Test module printing """
        experiment = hxsnn.Experiment()
        module = hxsnn.BatchDropout(33, 0.5, experiment)
        logger.INFO(module)

    def test_set_mask(self):
        """ Test mask is updated properly """
        experiment = hxsnn.Experiment()
        dropout = hxsnn.BatchDropout(33, 0.5, experiment)

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
        class BatchDropout(hxsnn.BatchDropout):
            def forward_func(self, x):
                return x, self.mask

        experiment = hxsnn.Experiment()
        dropout = BatchDropout(33, 0.5, experiment)
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
