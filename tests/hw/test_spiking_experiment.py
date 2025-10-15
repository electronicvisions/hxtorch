"""
Test HX Modules
"""
import unittest
import torch

import pygrenade_vx as grenade

import hxtorch
from hxtorch.core.utils import calib_helper
from hxtorch.spiking import Experiment
from hxtorch.spiking.modules import HXModule, LIF, Synapse
from hxtorch.spiking.handle import LIFObservables


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
        handle1 = LIFObservables()
        handle2 = LIFObservables()
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
        handle3 = LIFObservables()
        handle4 = LIFObservables()
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

    def test_run(self):
        """ Test hardware results are returned properly """
        # Mock mode
        experiment = Experiment(mock=True)
        # Modules
        module1 = Synapse(10, 10, experiment, lambda x: x)
        module2 = LIF(10, experiment, lambda x: x)
        # Forward
        input_handle = LIFObservables(spikes=torch.randn((10, 10, 10)))
        handle1 = module1(input_handle)
        module2(handle1)
        # Two modules should now be registered
        self.assertEqual(len(experiment.modules.nodes), 2)
        # Get results -> In mock there are no hardware results
        results = experiment.run(10)
        self.assertIsNone(results, None)
        # No input node should be injected
        self.assertEqual(len(experiment.modules.nodes), 2)
        # Do it again -> This should not change anything
        results = experiment.run(10)
        self.assertEqual(results, None)
        self.assertEqual(len(experiment.modules.nodes), 2)

        # HW mode
        experiment = Experiment(mock=False)
        experiment.calibration = calib_helper.fixture_calibration_from_file(
            calib_helper.nightly_calib_path()
        )

        # Modules
        module1 = Synapse(10, 10, experiment)
        module2 = LIF(10, experiment)
        # Forward
        input_handle = LIFObservables(spikes=torch.randn((10, 10, 10)))
        handle1 = module1(input_handle)
        module2(handle1)
        self.assertEqual(len(experiment.modules.nodes), 2)
        results = experiment.run(10)
        self.assertEqual(len(experiment.modules.nodes), 3)

        # Execute again -> should still work as expected, in training we also
        results = experiment.run(10)
        self.assertEqual(len(experiment.modules.nodes), 3)

        # Deeper net
        experiment = Experiment(mock=False)
        experiment.calibration = calib_helper.fixture_calibration_from_file(
            calib_helper.nightly_calib_path()
        )

        # Modules
        module1 = Synapse(10, 10, experiment)
        module2 = LIF(10, experiment)
        module3 = Synapse(10, 10, experiment)
        module4 = LIF(10, experiment)
        module5 = Synapse(10, 10, experiment)
        module6 = LIF(10, experiment)

        # Forward
        input_handle = LIFObservables(spikes=torch.randn((20, 10, 10)))
        handle1 = module1(input_handle)
        handle2 = module2(handle1)
        handle3 = module3(handle2)
        handle4 = module4(handle3)
        handle5 = module5(handle4)
        module6(handle5)

        # Six modules should now be registered
        self.assertEqual(len(experiment.modules.nodes), 6)
        results = experiment.run(20)
        self.assertEqual(len(experiment.modules.nodes), 7)

    def test_inter_batch_entry_wait(self):
        experiment = Experiment(mock=False)
        experiment.calibration = calib_helper.fixture_calibration_from_file(
            calib_helper.nightly_calib_path()
        )

        module1 = Synapse(10, 10, experiment)
        module2 = LIF(10, experiment)
        input_handle = LIFObservables(spikes=torch.randn((2, 10, 10)))
        handle1 = module1(input_handle)
        module2(handle1)

        inter_batch_entry_wait = int(250e6)
        experiment.inter_batch_entry_wait = inter_batch_entry_wait

        result = experiment.run(10)

        self.assertLess(
            int(inter_batch_entry_wait / 125),
            int(result.execution_instances.get(
                grenade.common.ExecutionInstanceOnExecutor()
                ).device_usage_duration[
                    grenade.common.ChipOnConnection()
                ].total_seconds() * 1e6,
            ),
        )

if __name__ == "__main__":
    unittest.main()
