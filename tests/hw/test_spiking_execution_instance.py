""" Test ExecutionInstance(s) """
import unittest
import hxtorch
import pylogging as logger
import pygrenade_vx as grenade

from hxtorch.spiking.execution_instance import (
    ExecutionInstance, ExecutionInstances)
from hxtorch.spiking.utils import calib_helper

logger.default_config(level=logger.LogLevel.INFO)
logger = logger.get("hxtorch.test.hw.test_spiking_execution_instance")


class HXTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()


class TestExecutionInstances(HXTestCase):

    def test_chips(self):
        """ Test chips are returned """
        inst1 = ExecutionInstance()
        inst2 = ExecutionInstance()
        inst3 = ExecutionInstance()
        instances = ExecutionInstances([inst1, inst2, inst3])
        self.assertEqual(len(instances), 3)
        chips = instances.chips
        self.assertEqual(len(chips), 3)
        self.assertListEqual(
            list(chips.values()), [None, None, None])
        self.assertTrue(
            all(inst in chips.keys()
                for inst in [inst1.ID, inst2.ID, inst3.ID]))
        self.assertTrue(
            all(inst in chips.values()
                for inst in [inst1.chip, inst2.chip, inst3.chip]))

        # load chips
        for inst in instances:
            inst.prepare_static_config()
        chips = instances.chips
        self.assertEqual(len(chips), 3)
        self.assertTrue(
            all(inst in chips.keys()
                for inst in [inst1.ID, inst2.ID, inst3.ID]))
        self.assertTrue(
            all(inst in chips.values()
                for inst in [inst1.chip, inst2.chip, inst3.chip]))
        self.assertTrue(
            all(chip is not None
                for chip in [inst1.chip, inst2.chip, inst3.chip]))

    def test_cadc_recordings(self):
        """ Test CADC recordings are returned """
        inst1 = ExecutionInstance()
        neurons1 = {
            0: [grenade.network.CADCRecording.Neuron()],
            1: [grenade.network.CADCRecording.Neuron(),
                grenade.network.CADCRecording.Neuron()]}
        inst1.cadc_neurons.update(neurons1)
        inst2 = ExecutionInstance()
        inst3 = ExecutionInstance()
        neurons3 = {
            0: [grenade.network.CADCRecording.Neuron()]}
        inst3.cadc_neurons.update(neurons3)

        instances = ExecutionInstances([inst1, inst2, inst3])
        cadc_recordings = instances.cadc_recordings
        self.assertIsInstance(cadc_recordings, dict)
        self.assertEqual(len(cadc_recordings), 2)
        for _id, recording in cadc_recordings.items():
            self.assertIsInstance(recording, grenade.network.CADCRecording)
            self.assertTrue(_id in [inst1.ID, inst2.ID, inst3.ID])
            inst = [inst for inst in [inst1, inst2, inst3]
                    if inst.ID == _id ].pop()
            self.assertListEqual(
                recording.neurons, inst.cadc_recordings().neurons)

    def test_playback_hooks(self):
        """ Test playback hooks are returned """
        inst1 = ExecutionInstance()
        inst2 = ExecutionInstance()
        inst3 = ExecutionInstance()
        instances = ExecutionInstances([inst1, inst2, inst3])
        for inst in instances:
            inst.prepare_static_config()

        hooks = instances.playback_hooks
        self.assertIsInstance(hooks, dict)
        self.assertEqual(len(hooks), 3)

        for _id, hook in hooks.items():
            self.assertIsInstance(
                hook,
                grenade.signal_flow.ExecutionInstanceHooks)
            self.assertTrue(_id in [inst1.ID, inst2.ID, inst3.ID])


class TestExecutionInstance(HXTestCase):

    def test_instance_id(self):
        """ Test instance ID creation """
        inst1 = ExecutionInstance()
        inst2 = ExecutionInstance()
        self.assertNotEqual(inst1.ID, inst2.ID)
        self.assertEqual(id(inst1), int(inst1.ID))

    def test_str(self):
        """ Test string representation of execution instance """
        inst1 = ExecutionInstance()
        inst2 = ExecutionInstance()
        logger.INFO(inst1)
        logger.INFO(inst2)

    def test_load_calib(self):
        """ Test calibration is loaded correctly """
        inst = ExecutionInstance()
        calib_path = calib_helper.nightly_calib_path()
        inst.load_calib(calib_path)
        self.assertIsNotNone(inst.chip)

    def test_prepare_static_config(self):
        """ Test prepare static config """
        # Test calib
        # Test chip is None
        inst = ExecutionInstance()
        self.assertIsNone(inst.chip)
        inst.prepare_static_config()
        self.assertIsNotNone(inst.chip)

        # Test chip is not None -> No new chip
        calib_path = calib_helper.nightly_calib_path()
        inst = ExecutionInstance(calib_path=calib_path)
        chip1 = inst.chip
        self.assertIsNotNone(inst.chip)
        inst.prepare_static_config()
        chip2 = inst.chip
        self.assertEqual(chip1, chip2)

    def test_cadc_recordings(self):
        """ Test CADC recordings """
        inst = ExecutionInstance()
        self.assertRaises(AssertionError, inst.cadc_recordings)

        inst = ExecutionInstance()
        self.assertEqual(len(inst.cadc_neurons), 0)
        neurons = {
            0: [grenade.network.CADCRecording.Neuron()],
            1: [grenade.network.CADCRecording.Neuron(),
                grenade.network.CADCRecording.Neuron()]}
        inst.cadc_neurons.update(neurons)
        self.assertEqual(len(inst.cadc_neurons), 2)
        cadc_recordings = inst.cadc_recordings()
        self.assertIsInstance(cadc_recordings, grenade.network.CADCRecording)
        self.assertEqual(len(cadc_recordings.neurons), 2)
        self.assertEqual(
            cadc_recordings.neurons, [nrn[0] for nrn in neurons.values()])

    def test_generate_playback_hooks(self):
        """ Test generate playback hooks """
        inst = ExecutionInstance()
        hooks = inst.generate_playback_hooks()
        self.assertIsInstance(hooks, grenade.signal_flow.ExecutionInstanceHooks)


if __name__ == "__main__":
    unittest.main()