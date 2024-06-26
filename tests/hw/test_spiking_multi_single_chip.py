"""
Test HX Modules
"""
import unittest
import torch

import hxtorch
from hxtorch.spiking import Experiment, run
from hxtorch.spiking.modules import InputNeuron, Neuron, Synapse
from hxtorch.spiking.handle import NeuronHandle
from pygrenade_vx.common import ExecutionInstanceID


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
        module1 = Synapse(10, 10, experiment,
                          execution_instance=ExecutionInstanceID(0))
        module2 = Neuron(10, experiment,
                         execution_instance=ExecutionInstanceID(0))
        # switch execution instance
        module3 = Synapse(10, 10, experiment,
                          execution_instance=ExecutionInstanceID(1))
        module4 = Neuron(10, experiment,
                         execution_instance=ExecutionInstanceID(1))

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
        synapse1 = Synapse(10, 8, experiment,
                           execution_instance=ExecutionInstanceID(0))
        synapse2 = Synapse(10, 8, experiment,
                           execution_instance=ExecutionInstanceID(1))
        neuron1 = Neuron(8, experiment,
                         execution_instance=ExecutionInstanceID(0))
        neuron2 = Neuron(8, experiment,
                         execution_instance=ExecutionInstanceID(1))

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

        # Modules
        module1 = Synapse(10, 10, experiment,
                          execution_instance=ExecutionInstanceID(0))
        module2 = Neuron(10, experiment,
                         execution_instance=ExecutionInstanceID(0))
        # switch execution instance
        module3 = Synapse(10, 10, experiment,
                          execution_instance=ExecutionInstanceID(1))
        module4 = Neuron(10, experiment,
                         execution_instance=ExecutionInstanceID(1))

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
