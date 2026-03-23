"""
Test infrastructure for event drops
"""
import unittest

import torch

import hxtorch.spiking as hxsnn


# pylint: disable=too-many-instance-attributes
class TestSynapseMockMode(unittest.TestCase):
    """
    Test infrastructure for pre-synaptic event drops of the `AELIF` and
    `Synapse` modules.
    """
    def setUp(self):
        self.population_size = 1
        self.batch_size = 1
        self.time_steps = 100
        self.dt = 1e-6

        self.inputs = torch.ones(self.time_steps, self.batch_size, 1)
        self.weights = torch.nn.parameter.Parameter(torch.tensor([[100.]]))

        # Define event drop transforms that simply delete all events
        self.event_drop_transform_synapse = \
            lambda events: torch.zeros_like(events)

        def event_drop_transform_neuron(spikes: torch.Tensor) -> None:
            spikes.zero_()
            return
        self.event_drop_transform_neuron = event_drop_transform_neuron


    def test_synapse_event_drops(self):
        """
        Test infrastructure for pre-synaptic event drops of the `Synapse`
        module.
        Run a 2-layer network once without event drops and once with all events
        being erased in the second synapse layer. (Simulates event drops
        happening in event transport from neuron circuits to synapse drivers)
        Between those two runs, the events of the first neuron layer should
        look the same, whereas the events of the second neuron layer are
        expected to differ.
        """
        # Run first experiment, without event drops (=ideal)
        exp_ideal = hxsnn.Experiment(mock=True)

        # Layers
        syn1 = hxsnn.modules.Synapse(1, 1, experiment=exp_ideal)
        syn1.weight.data = self.weights
        syn2 = hxsnn.modules.Synapse(1, 1, experiment=exp_ideal)
        syn2.weight.data = self.weights
        lif1 = hxsnn.modules.LIF(
            self.population_size, dt=self.dt, experiment=exp_ideal)
        lif2 = hxsnn.modules.LIF(
            self.population_size, dt=self.dt, experiment=exp_ideal)

        # Forward
        syn1_handle = syn1(hxsnn.LIFObservables(spikes=self.inputs))
        lif1_handle = lif1(syn1_handle)
        syn2_handle = syn2(lif1_handle)
        lif2_handle = lif2(syn2_handle)
        hxsnn.run(exp_ideal, self.time_steps)

        # Save spike tensors of neuron modules
        lif1_spikes_ideal = lif1_handle.spikes.detach().clone()
        lif2_spikes_ideal = lif2_handle.spikes.detach().clone()

        # Run second experiment, with event drops
        exp_drops = hxsnn.Experiment(mock=True)

        # Layers
        syn1 = hxsnn.modules.Synapse(1, 1, experiment=exp_drops)
        syn1.weight.data = self.weights
        syn2 = hxsnn.modules.Synapse(
            1, 1, experiment=exp_drops,
            event_drop_transform=self.event_drop_transform_synapse)
        syn2.weight.data = self.weights
        lif1 = hxsnn.modules.LIF(
            self.population_size, dt=self.dt, experiment=exp_drops)
        lif2 = hxsnn.modules.LIF(
            self.population_size, dt=self.dt, experiment=exp_drops)

        # Forward
        syn1_handle = syn1(hxsnn.LIFObservables(spikes=self.inputs))
        lif1_handle = lif1(syn1_handle)
        syn2_handle = syn2(lif1_handle)
        lif2_handle = lif2(syn2_handle)
        hxsnn.run(exp_drops, self.time_steps)

        # Save spike tensors of neuron modules
        lif1_spikes_drops = lif1_handle.spikes.detach().clone()
        lif2_spikes_drops = lif2_handle.spikes.detach().clone()

        # Expect the same events in both runs in the first neuron layer
        self.assertTrue(torch.allclose(lif1_spikes_ideal, lif1_spikes_drops))
        # Expect event drops to affect second neuron layer
        self.assertFalse(torch.allclose(lif2_spikes_ideal, lif2_spikes_drops))


    def test_neuron_event_drops(self):
        """
        Test infrastructure for pre-synaptic event drops of the `AELIF` module.
        Run a 2-layer network once without event drops and once with all events
        being erased from the spike tensor of the first neuron layer after
        simulation. (Simulates event drops during recording / readout)
        Between those two runs, the events of the first neuron layer should
        differ, whereas the events of the second neuron layer are expected to
        look the same.
        """
        # Run first experiment, without event drops (=ideal)
        exp_ideal = hxsnn.Experiment(mock=True)

        # Layers
        syn1 = hxsnn.modules.Synapse(1, 1, experiment=exp_ideal)
        syn1.weight.data = self.weights
        syn2 = hxsnn.modules.Synapse(1, 1, experiment=exp_ideal)
        syn2.weight.data = self.weights
        lif1 = hxsnn.modules.LIF(
            self.population_size, dt=self.dt, experiment=exp_ideal)
        lif2 = hxsnn.modules.LIF(
            self.population_size, dt=self.dt, experiment=exp_ideal)

        # Forward
        syn1_handle = syn1(hxsnn.LIFObservables(spikes=self.inputs))
        lif1_handle = lif1(syn1_handle)
        syn2_handle = syn2(lif1_handle)
        lif2_handle = lif2(syn2_handle)
        hxsnn.run(exp_ideal, self.time_steps)

        # Save spike tensors of neuron modules
        lif1_spikes_ideal = lif1_handle.spikes.detach().clone()
        lif2_spikes_ideal = lif2_handle.spikes.detach().clone()

        # Run second experiment, with event drops
        exp_drops = hxsnn.Experiment(mock=True)

        # Layers
        syn1 = hxsnn.modules.Synapse(1, 1, experiment=exp_drops)
        syn1.weight.data = self.weights
        syn2 = hxsnn.modules.Synapse(1, 1, experiment=exp_drops)
        syn2.weight.data = self.weights
        lif1 = hxsnn.modules.LIF(
            self.population_size, dt=self.dt, experiment=exp_drops,
            event_drop_transform=self.event_drop_transform_neuron)
        lif2 = hxsnn.modules.LIF(
            self.population_size, dt=self.dt, experiment=exp_drops)

        # Forward
        syn1_handle = syn1(hxsnn.LIFObservables(spikes=self.inputs))
        lif1_handle = lif1(syn1_handle)
        syn2_handle = syn2(lif1_handle)
        lif2_handle = lif2(syn2_handle)
        hxsnn.run(exp_drops, self.time_steps)

        # Save spike tensors of neuron modules
        lif1_spikes_drops = lif1_handle.spikes.detach().clone()
        lif2_spikes_drops = lif2_handle.spikes.detach().clone()

        # Expect events to differ between both runs in the first neuron layer
        self.assertFalse(torch.allclose(lif1_spikes_ideal, lif1_spikes_drops))
        # Expect event drops durint readout to not affect second neuron layer
        self.assertTrue(torch.allclose(lif2_spikes_ideal, lif2_spikes_drops))


if __name__ == "__main__":
    unittest.main()
