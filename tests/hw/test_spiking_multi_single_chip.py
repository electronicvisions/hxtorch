"""
Test HX Modules
"""
import unittest
import torch

import hxtorch
from hxtorch.spiking import Experiment, run
from hxtorch.spiking.modules import Neuron, Synapse
from hxtorch.spiking.handle import NeuronHandle
from hxtorch.spiking.utils import calib_helper
from hxtorch.spiking.execution_instance import ExecutionInstance


class TestMultiSingleChip(unittest.TestCase):
    """ Test multi-single-chip experiment """

    def setUp(cls):
        hxtorch.init_hardware()

    def tearDown(cls):
        hxtorch.release_hardware()

    def test_feedforward(self):
        """
        Test inter-execution-instance connections are created correctly.
        """
        experiment = Experiment(mock=False)

        # Modules
        instance1 = ExecutionInstance(
            calib_path=calib_helper.nightly_calib_path())
        module1 = Synapse(10, 5, experiment, execution_instance=instance1)
        module2 = Neuron(
            5, experiment, execution_instance=instance1,
            enable_cadc_recording=True)

        # switch execution instance
        instance2 = ExecutionInstance(
            calib_path=calib_helper.nightly_calib_path())
        module3 = Synapse(5, 15, experiment, execution_instance=instance2)
        module4 = Neuron(
            15, experiment, execution_instance=instance2,
            enable_cadc_recording=True)

        # Forward
        input_handle = NeuronHandle(spikes=torch.randn((20, 10, 10)))
        handle1 = module1(input_handle)
        handle2 = module2(handle1)
        handle3 = module3(handle2)
        module4(handle3)

        # Only test that execution works
        run(experiment, 10)

    def test_feedforward_multiple_executions(self):
        """ Test multi-single-chip experiment with multiple runs """

        experiment = Experiment(mock=False)
        inst1 = ExecutionInstance(calib_path=calib_helper.nightly_calib_path())
        inst2 = ExecutionInstance(calib_path=calib_helper.nightly_calib_path())
        synapse1 = Synapse(10, 8, experiment, execution_instance=inst1)
        synapse2 = Synapse(10, 8, experiment, execution_instance=inst2)
        neuron1 = Neuron(8, experiment, execution_instance=inst1)
        neuron2 = Neuron(8, experiment, execution_instance=inst2)

        # Forward 1
        inputs = torch.randn((10, 10, 10))
        neuron1(synapse1(NeuronHandle(inputs)))
        neuron2(synapse2(NeuronHandle(inputs)))
        run(experiment, 10)

        # Forward 2
        inputs = torch.randn((10, 10, 10))
        neuron1(synapse1(NeuronHandle(inputs)))
        neuron2(synapse2(NeuronHandle(inputs)))
        run(experiment, 10)


    def test_feedforward_multiple_inputs(self):
        """
        Test inter-execution-instance connections are created correctly with
        multiple inputs.
        """
        experiment = Experiment(mock=False)
        inst1 = ExecutionInstance(calib_path=calib_helper.nightly_calib_path())
        inst2 = ExecutionInstance(calib_path=calib_helper.nightly_calib_path())

        # Modules
        module1 = Synapse(10, 10, experiment, execution_instance=inst1)
        module2 = Neuron(10, experiment, execution_instance=inst1)
        # switch execution instance
        module3 = Synapse(10, 10, experiment, execution_instance=inst2)
        module4 = Neuron(10, experiment, execution_instance=inst2)

        # Forward
        input_handle = NeuronHandle(spikes=torch.randn((20, 10, 10)))
        input_handle2 = NeuronHandle(spikes=torch.randn((20, 10, 10)))
        handle1 = module1(input_handle)
        module2(handle1)
        handle3 = module3(input_handle2)
        module4(handle3)

        # Only test that execution works
        run(experiment, 10)


if __name__ == "__main__":
    unittest.main()
