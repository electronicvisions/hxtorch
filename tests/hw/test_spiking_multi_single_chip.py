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


if __name__ == "__main__":
    unittest.main()
