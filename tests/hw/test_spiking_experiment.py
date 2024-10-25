"""
Test HX Modules
"""
import unittest
import torch

import hxtorch
from hxtorch.spiking import Experiment
from hxtorch.spiking.modules import HXModule, InputNeuron, Neuron, Synapse
from hxtorch.spiking.handle import NeuronHandle
from dlens_vx_v3.halco import DLSGlobal


class TestExperiment(unittest.TestCase):
    """ Test Experiment """

    def setUp(cls):
        hxtorch.init_hardware()

    def tearDown(cls):
        hxtorch.release_hardware()

    def test_connect(self):
        """
        Test connections are created correctly.
        """
        experiment = Experiment(mock=True)

        # Add one connection
        module1 = HXModule(experiment, None)
        handle1 = NeuronHandle()
        handle2 = NeuronHandle()
        experiment.connect(module1, handle1, handle2)

        # Check connection is registered
        sources = [
            e["handle"] for _, _, e in experiment.modules.graph.in_edges(
                experiment.modules.nodes[module1], data=True)]
        targets = [
            e["handle"] for _, _, e in experiment.modules.graph.out_edges(
                experiment.modules.nodes[module1], data=True)]
        self.assertEqual(len(sources), 1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(sources[0], handle1)
        self.assertEqual(targets[0], handle2)

        # Add another one
        module2 = HXModule(experiment, None)
        handle3 = NeuronHandle()
        handle4 = NeuronHandle()
        experiment.connect(module2, handle3, handle4)
        # Check connection is registered
        sources = [
            e["handle"] for _, _, e in experiment.modules.graph.in_edges(
                experiment.modules.nodes[module2], data=True)]
        targets = [
            e["handle"] for _, _, e in experiment.modules.graph.out_edges(
                experiment.modules.nodes[module2], data=True)]
        self.assertEqual(len(sources), 1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(sources[0], handle3)
        self.assertEqual(targets[0], handle4)

        # There should be two connections present now
        self.assertEqual(len(experiment.modules.nodes), 2)

    def test_get_hw_result(self):
        """ Test hardware results are returned properly """
        # Mock mode
        experiment = Experiment(mock=True)
        # Modules
        module1 = Synapse(10, 10, experiment, lambda x: x)
        module2 = Neuron(10, experiment, lambda x: x)
        # Forward
        input_handle = NeuronHandle(spikes=torch.randn((10, 10, 10)))
        handle1 = module1(input_handle)
        module2(handle1)
        # Two modules should now be registered
        self.assertEqual(len(experiment.modules.nodes), 2)
        # Get results -> In mock there are no hardware results
        results = experiment.get_hw_results(10)
        self.assertEqual(results, (dict(), None))
        # No input node should be injected
        self.assertEqual(len(experiment.modules.nodes), 2)
        # Do it again -> This should not change anything
        results = experiment.get_hw_results(10)
        self.assertEqual(results, (dict(), None))
        self.assertEqual(len(experiment.modules.nodes), 2)

        # HW mode
        experiment = Experiment(mock=False)
        # Modules
        module1 = Synapse(10, 10, experiment)
        module2 = Neuron(10, experiment)
        # Forward
        input_handle = NeuronHandle(spikes=torch.randn((10, 10, 10)))
        handle1 = module1(input_handle)
        module2(handle1)
        self.assertEqual(len(experiment.modules.nodes), 2)
        results, _ = experiment.get_hw_results(10)
        self.assertEqual(len(experiment.modules.nodes), 3)
        self.assertEqual(len(experiment._populations), 2)
        self.assertEqual(len(experiment._projections), 1)
        for pop in experiment._populations:
            if isinstance(pop, InputNeuron):
                continue
            self.assertIsNotNone(results.get(pop.descriptor))

        # Execute again -> should still work as expected, in training we also
        results, _ = experiment.get_hw_results(10)
        self.assertEqual(len(experiment.modules.nodes), 3)
        self.assertEqual(len(experiment._populations), 2)
        self.assertEqual(len(experiment._projections), 1)
        for pop in experiment._populations:
            if isinstance(pop, InputNeuron):
                continue
            self.assertIsNotNone(results.get(pop.descriptor))

        # Deeper net
        experiment = Experiment(mock=False)
        # Modules
        module1 = Synapse(10, 10, experiment)
        module2 = Neuron(10, experiment)
        module3 = Synapse(10, 10, experiment)
        module4 = Neuron(10, experiment)
        module5 = Synapse(10, 10, experiment)
        module6 = Neuron(10, experiment)

        # Forward
        input_handle = NeuronHandle(spikes=torch.randn((20, 10, 10)))
        handle1 = module1(input_handle)
        handle2 = module2(handle1)
        handle3 = module3(handle2)
        handle4 = module4(handle3)
        handle5 = module5(handle4)
        module6(handle5)

        # Six modules should now be registered
        self.assertEqual(len(experiment.modules.nodes), 6)
        results, _ = experiment.get_hw_results(20)
        self.assertEqual(len(experiment.modules.nodes), 7)
        self.assertEqual(len(experiment._populations), 4)
        self.assertEqual(len(experiment._projections), 3)
        for pop in experiment._populations:
            if isinstance(pop, InputNeuron):
                continue
            self.assertIsNotNone(results.get(pop.descriptor))

    def test_inter_batch_entry_wait(self):
        experiment = Experiment(mock=False)
        module1 = Synapse(10, 10, experiment)
        module2 = Neuron(10, experiment)
        input_handle = NeuronHandle(spikes=torch.randn((2, 10, 10)))
        handle1 = module1(input_handle)
        module2(handle1)

        inter_batch_entry_wait = int(250e6)
        experiment.inter_batch_entry_wait = inter_batch_entry_wait

        _, times = experiment.get_hw_results(10)

        self.assertLess(int(inter_batch_entry_wait / 125), int(times.execution_duration_per_hardware[DLSGlobal()].total_seconds()*1e6))

if __name__ == "__main__":
    unittest.main()
