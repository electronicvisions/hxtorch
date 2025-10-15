"""
Test HX Modules
"""
import unittest
import torch

from dlens_vx_v3 import halco
import pygrenade_vx as grenade

import hxtorch
from hxtorch.spiking import Experiment, run
from hxtorch.spiking.modules import LIF, Synapse
from hxtorch.spiking.handle import LIFObservables
from hxtorch.core.utils import calib_helper


class TestMultiChipExperiments(unittest.TestCase):
    """ Test multi-single-chip experiment """

    def setUp(cls):
        hxtorch.init_hardware()

    def tearDown(cls):
        hxtorch.release_hardware()

    @unittest.skip("Test works, but takes too long")
    def test_feedforward_one_layer(self):
        """
        Test inter-execution-instance connections are created correctly.
        """
        experiment = Experiment(mock=False)

        calibration = grenade.network.abstract.FixtureCalibration()
        calibration.chips = {
            grenade.common.ExecutionInstanceOnExecutor(
                grenade.common.ExecutionInstanceID(0),
                grenade.common.ConnectionOnExecutor(0)
            ): {
                grenade.common.ChipOnConnection():
                calib_helper.chip_from_file(calib_helper.nightly_calib_path())
            },
            grenade.common.ExecutionInstanceOnExecutor(
                grenade.common.ExecutionInstanceID(1),
                grenade.common.ConnectionOnExecutor(0)
            ): {
                grenade.common.ChipOnConnection():
                calib_helper.chip_from_file(calib_helper.nightly_calib_path())
            }
        }
        experiment.calibration = calibration

        # Modules
        module1 = Synapse(128, 1024, experiment)
        module2 = LIF(1024, experiment, enable_cadc_recording=False)

        # Forward
        input_handle = LIFObservables(spikes=torch.zeros((20, 10, 128)))
        handle1 = module1(input_handle)
        module2(handle1)

        # Only test that execution works
        run(experiment, 10)

    @unittest.skip("Test works, but takes too long")
    def test_feedforward_multiple_layers(self):
        """
        Test inter-execution-instance connections are created correctly.
        """
        experiment = Experiment(mock=False)
        calibration = grenade.network.abstract.FixtureCalibration()
        calibration.chips = {
            grenade.common.ExecutionInstanceOnExecutor(
                grenade.common.ExecutionInstanceID(0),
                grenade.common.ConnectionOnExecutor(0)
            ): {
                grenade.common.ChipOnConnection():
                calib_helper.chip_from_file(calib_helper.nightly_calib_path())
            },
            grenade.common.ExecutionInstanceOnExecutor(
                grenade.common.ExecutionInstanceID(1),
                grenade.common.ConnectionOnExecutor(0)
            ): {
                grenade.common.ChipOnConnection():
                calib_helper.chip_from_file(calib_helper.nightly_calib_path())
            },
            grenade.common.ExecutionInstanceOnExecutor(
                grenade.common.ExecutionInstanceID(2),
                grenade.common.ConnectionOnExecutor(0)
            ): {
                grenade.common.ChipOnConnection():
                calib_helper.chip_from_file(calib_helper.nightly_calib_path())
            },
            grenade.common.ExecutionInstanceOnExecutor(
                grenade.common.ExecutionInstanceID(3),
                grenade.common.ConnectionOnExecutor(0)
            ): {
                grenade.common.ChipOnConnection():
                calib_helper.chip_from_file(calib_helper.nightly_calib_path())
            },
            grenade.common.ExecutionInstanceOnExecutor(
                grenade.common.ExecutionInstanceID(4),
                grenade.common.ConnectionOnExecutor(0)
            ): {
                grenade.common.ChipOnConnection():
                calib_helper.chip_from_file(calib_helper.nightly_calib_path())
            },
        }
        experiment.calibration = calibration

        experiment.mapper.neuron_permutation = [
            halco.AtomicNeuronOnDLS(halco.common.Enum(i))
            for i in range(128)
        ] + [
            halco.AtomicNeuronOnDLS(halco.common.Enum(i))
            for i in range(256, 384)
        ]

        # Modules
        module1 = Synapse(128, 128, experiment)
        module2 = LIF(128, experiment, enable_cadc_recording=False)
        module3 = Synapse(128, 128, experiment)
        module4 = LIF(128, experiment, enable_cadc_recording=False)
        module5 = Synapse(128, 128, experiment)
        module6 = LIF(128, experiment, enable_cadc_recording=False)
        module7 = Synapse(128, 128, experiment)
        module8 = LIF(128, experiment, enable_cadc_recording=False)
        module9 = Synapse(128, 128, experiment)
        module10 = LIF(128, experiment, enable_cadc_recording=False)
        module11 = Synapse(128, 128, experiment)
        module12 = LIF(128, experiment, enable_cadc_recording=False)

        # Forward
        input_handle = LIFObservables(spikes=torch.randn((20, 10, 128)))
        handle1 = module1(input_handle)
        handle2 = module2(handle1)
        handle3 = module3(handle2)
        handle4 = module4(handle3)
        handle5 = module5(handle4)
        handle6 = module6(handle5)
        handle7 = module7(handle6)
        handle8 = module8(handle7)
        handle9 = module9(handle8)
        handle10 = module10(handle9)
        handle11 = module11(handle10)
        module12(handle11)

        # Only test that execution works
        run(experiment, 10)

    def test_feedforward_multiple_inputs(self):
        """
        Test inter-execution-instance connections are created correctly with
        multiple inputs.
        """
        experiment = Experiment(mock=False)
        calibration = grenade.network.abstract.FixtureCalibration()
        calibration.chips = {
            grenade.common.ExecutionInstanceOnExecutor(
                grenade.common.ExecutionInstanceID(0),
                grenade.common.ConnectionOnExecutor(0)
            ): {
                grenade.common.ChipOnConnection():
                calib_helper.chip_from_file(calib_helper.nightly_calib_path())
            },
            grenade.common.ExecutionInstanceOnExecutor(
                grenade.common.ExecutionInstanceID(1),
                grenade.common.ConnectionOnExecutor(0)
            ): {
                grenade.common.ChipOnConnection():
                calib_helper.chip_from_file(calib_helper.nightly_calib_path())
            }
        }
        experiment.calibration = calibration

        # Modules
        module1 = Synapse(10, 512, experiment)
        module2 = LIF(512, experiment, enable_cadc_recording=False)
        # switch execution instance
        module3 = Synapse(10, 512, experiment)
        module4 = LIF(512, experiment, enable_cadc_recording=False)

        # Forward
        input_handle = LIFObservables(spikes=torch.randn((20, 10, 10)))
        input_handle2 = LIFObservables(spikes=torch.randn((20, 10, 10)))
        handle1 = module1(input_handle)
        module2(handle1)
        handle3 = module3(input_handle2)
        module4(handle3)

        # Only test that execution works
        run(experiment, 10)


if __name__ == "__main__":
    unittest.main()
