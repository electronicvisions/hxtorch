"""
Test snn.HXNeuron
"""
import unittest

import torch

import hxtorch
from hxtorch import spiking as hxsnn


hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)
logger = hxtorch.logger.get("hxtorch.test.hw.test_spiking_modules")


class TestHXModuleWrapper(unittest.TestCase):
    """ TEst HXModuleWrapper """

    def test_contains(self):
        """ Test wrapper contains module """
        # Experiment
        experiment = hxsnn.Experiment()

        # Modules
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        lif = hxsnn.LIF(10, experiment=experiment)

        wrapper = hxsnn.HXModuleWrapper(experiment, linear=linear, lif=lif)

        # Should contain
        self.assertTrue(wrapper.contains(linear))
        self.assertTrue(wrapper.contains(lif))
        self.assertTrue(wrapper.contains([linear, lif]))

        # Should not contain
        lif2 = hxsnn.LIF(10, experiment=experiment)
        self.assertFalse(wrapper.contains(lif2))
        self.assertFalse(wrapper.contains([lif2]))
        self.assertFalse(wrapper.contains([linear, lif2]))

    def test_print(self):
        """ Test module printing """
        experiment = hxsnn.Experiment()
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        lif = hxsnn.LIF(10, experiment=experiment)
        module = hxsnn.HXModuleWrapper(experiment, linear=linear, lif=lif)
        logger.INFO(module)

    def test_update(self):
        """ Test update modules """
        # Experiment
        experiment = hxsnn.Experiment()

        # Modules
        linear1 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif1 = hxsnn.LIF(10, experiment=experiment)
        wrapper = hxsnn.HXModuleWrapper(experiment, linear1=linear1, lif1=lif1)
        self.assertEqual({"linear1": linear1, "lif1": lif1}, wrapper.modules)

        linear2 = hxsnn.Synapse(10, 10, experiment=experiment)
        lif2 = hxsnn.LIF(10, experiment=experiment)
        wrapper.update(linear1=linear2, lif1=lif2)
        self.assertEqual({"linear1": linear2, "lif1": lif2}, wrapper.modules)

    def test_exec_forward(self):
        """ """
        in_tensor = torch.zeros(10, 5, 10)

        class Wrapper(hxsnn.HXModuleWrapper):
            def forward_func(selfw, input, hw_data=None):
                return (
                    hxsnn.Handle(res="res1"),
                    hxsnn.Handle(res="res2"),
                    hxsnn.Handle(res="res3"),
                    hxsnn.Handle(res="res4"))

        class Module(hxsnn.HXModule):
            output_type = type(hxsnn.Handle('res'))
            def post_process(selfw, hw_data, runtime):
                self.assertEqual(runtime, 0.001)
                return hw_data

        # Experiment
        experiment = hxsnn.Experiment()
        experiment.runtime_in_s = 0.001

        # Modules
        linear1 = Module(experiment)
        lif1 = Module(experiment)
        linear2 = Module(experiment)
        lif2 = Module(experiment)

        wrapper = Wrapper(
            experiment, linear1=linear1, lif1=lif1, linear2=linear2, lif2=lif2)

        # Forward
        in_h = hxsnn.LIFObservables(spikes=in_tensor)
        syn1 = linear1(in_h)
        nrn1 = lif1(syn1)
        syn2 = linear2(nrn1)
        nrn2 = lif2(syn2)

        inputs = (in_h,)
        outputs = (syn1, nrn1, syn2, nrn2)

        # Execute forward
        wrapper.exec_forward(inputs, outputs)

        self.assertEqual(syn1.res, "res1")
        self.assertEqual(nrn1.res, "res2")
        self.assertEqual(syn2.res, "res3")
        self.assertEqual(nrn2.res, "res4")

        # Test with HW data
        class HWDataWrapper(hxsnn.HXModuleWrapper):
            def forward_func(selfw, input, hw_data=None):
                self.assertEqual(
                    hw_data, (("syn1",), ("nrn1",), ("syn2",), ("nrn2",)))
                return (
                    hxsnn.Handle(res="res1"),
                    hxsnn.Handle(res="res2"),
                    hxsnn.Handle(res="res3"),
                    hxsnn.Handle(res="res4"))

        # Experiment
        experiment = hxsnn.Experiment(mock=False)
        experiment.runtime_in_s = 0.001

        # Modules
        linear1 = Module(experiment)
        lif1 = Module(experiment)
        linear2 = Module(experiment)
        lif2 = Module(experiment)

        wrapper = HWDataWrapper(
            experiment, linear1=linear1, lif1=lif1, linear2=linear2, lif2=lif2)

        # Forward
        in_h = hxsnn.LIFObservables(spikes=in_tensor)
        syn1 = linear1(in_h)
        nrn1 = lif1(syn1)
        syn2 = linear2(nrn1)
        nrn2 = lif2(syn2)

        inputs = (in_h,)
        outputs = (syn1, nrn1, syn2, nrn2)
        linear1.hw_observables = ("syn1",)
        lif1.hw_observables = ("nrn1",)
        linear2.hw_observables = ("syn2",)
        lif2.hw_observables = ("nrn2",)

        # Execute forward
        wrapper.exec_forward(inputs, outputs)

        self.assertEqual(syn1.res, "res1")
        self.assertEqual(nrn1.res, "res2")
        self.assertEqual(syn2.res, "res3")
        self.assertEqual(nrn2.res, "res4")

    def test_forward(self):
        in_tensor = torch.zeros(10, 5, 10)

        # Test with HW data
        class Wrapper(hxsnn.HXModuleWrapper):
            def forward_func(selfw, input, hw_data=None):
                self.assertEqual((("syn",), ("nrn",)))
                return (
                    hxsnn.SynapseHandle(graded_spikes="syn"),
                    hxsnn.LIFObservables(spikes="z1", membrane_cadc="v1"))

        # Experiment
        experiment = hxsnn.Experiment()

        # Modules
        linear = hxsnn.Synapse(10, 10, experiment=experiment)
        lif = hxsnn.LIF(10, experiment=experiment)
        wrapper = Wrapper(experiment, linear=linear, lif=lif)

        # forward
        inputs = hxsnn.LIFObservables()
        syn = linear(inputs)
        nrn = lif(syn)
        wrapper()

        self.assertEqual(len(experiment.modules.wrappers), 1)
        self.assertEqual(experiment.modules.wrappers, {wrapper: "w_0"})

        # forward again
        inputs = hxsnn.LIFObservables(spikes=in_tensor)
        syn = linear(inputs)
        nrn = lif(syn)
        wrapper()

        self.assertEqual(len(experiment.modules.wrappers), 1)
        self.assertEqual(experiment.modules.wrappers, {wrapper: "w_0"})


if __name__ == "__main__":
    unittest.main()
