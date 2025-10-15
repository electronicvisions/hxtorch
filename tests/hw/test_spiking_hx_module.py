"""
Test snn.HXNeuron
"""
import unittest
import torch

import hxtorch
from hxtorch import spiking as hxsnn


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
                if selfw.experiment.mock:
                    self.assertIsNone(hw_data)
                else:
                    self.assertEqual(hw_data, "hw_result")
                return input
            def post_process(selfw, hw_data, runtime):
                self.assertEqual(runtime, 0.001)
                return hw_data

        experiment = hxsnn.Experiment(mock=True)
        module = Module(experiment)
        module.param = "param1"

        # Input and output handles
        input_handle = hxsnn.LIFObservables(spikes=torch.zeros(10, 5))
        output_handle = hxsnn.LIFObservables()

        # Execute
        module.hw_observables.spikes = "hw_result"
        module.exec_forward(input_handle, output_handle)
        self.assertTrue(torch.equal(input_handle.spikes, output_handle.spikes))

        # HW
        experiment = hxsnn.Experiment(mock=False)
        experiment.runtime_in_s = 0.001
        module = Module(experiment)
        module.param = "param1"

        # Input and output handles
        input_handle = hxsnn.LIFObservables(spikes=torch.zeros(10, 5))
        output_handle = hxsnn.LIFObservables()

        # Execute
        module.hw_observables = "hw_result"
        module.exec_forward(input_handle, output_handle)
        self.assertTrue(torch.equal(input_handle.spikes, output_handle.spikes))


if __name__ == "__main__":
    unittest.main()
