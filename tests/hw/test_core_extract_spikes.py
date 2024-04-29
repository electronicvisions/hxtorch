"""
Test snn run function
"""
import unittest
from dlens_vx_v3 import halco, lola, hal

import hxtorch
import hxtorch.snn
import _hxtorch_core
import _hxtorch_spiking
import pygrenade_vx as grenade
from pygrenade_vx.network import (
    Population, ExternalSourcePopulation, Projection, Receptor)


class TestExtractNSpikes(unittest.TestCase):
    """ Test extract n spikes """

    batch_size: int = 3
    int_pop_size: int = 3
    ext_pop_size: int = 3

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()

    def generate_network(self, config):
        # Builder
        network_builder = grenade.network.NetworkBuilder()

        # Populations
        coords = []
        atomic_neurons = [an for an in halco.iter_all(halco.AtomicNeuronOnDLS)]
        morphology = hxtorch.snn.morphology.SingleCompartmentNeuron(1)
        for idx in range(self.int_pop_size):
            anchor = atomic_neurons[idx]
            ln = halco.LogicalNeuronOnDLS(morphology.compartments, anchor)
            coords.append(ln)
        # configure neurons
        for coord in coords:
            morphology.implement_morphology(coord, config[
                grenade.common.ExecutionInstanceID()].neuron_block)
            morphology.set_spike_recording(True, coord, config[
                grenade.common.ExecutionInstanceID()].neuron_block)
        neurons = [
            Population.Neuron(
                logical_neuron,
                {halco.CompartmentOnLogicalNeuron():
                 Population.Neuron.Compartment(
                     Population.Neuron.Compartment.SpikeMaster(0, True),
                     [set([Receptor(Receptor.ID(), Receptor.Type.excitatory),
                           Receptor(Receptor.ID(), Receptor.Type.inhibitory)])]
                    * len(logical_neuron.get_atomic_neurons()))})
            for i, logical_neuron in enumerate(coords)]
        int_pop = Population(neurons)
        ext_pop = ExternalSourcePopulation(
            [ExternalSourcePopulation.Neuron(False)] * self.ext_pop_size)
        self.int_pop_descr = network_builder.add(int_pop)
        self.ext_pop_descr = network_builder.add(ext_pop)
        # Some connections
        connections = []
        for i in range(self.ext_pop_size):
            connections.append(
                Projection.Connection(
                    (i, halco.CompartmentOnLogicalNeuron()),
                    (i, halco.CompartmentOnLogicalNeuron()),
                    Projection.Connection.Weight(63)))
        proj = Projection(
            Receptor(Receptor.ID(), Receptor.Type.excitatory),
            connections, self.ext_pop_descr, self.int_pop_descr)
        network_builder.add(proj)
        # Build network graph
        network = network_builder.done()
        routing_result = \
            grenade.network.routing.PortfolioRouter()(network)
        network_graph = \
            grenade.network.build_network_graph(network, routing_result)
        return network_graph, config

    def generate_inputs(self, network_graph):
        # Inputs
        input_generator = grenade.network.InputGenerator(
            network_graph, self.batch_size)
        # Add inputs
        times = []
        for b in range(self.batch_size):
            b_times = []
            for idx in range(self.ext_pop_size):
                b_times.append(
                    [(idx * 10 + 10) * int(
                        hal.Timer.Value.fpga_clock_cycles_per_us)])
            times.append(b_times)
        input_generator.add(times, self.ext_pop_descr)
        inputs = input_generator.done()
        # Add runtime
        inputs.runtime = [{
            grenade.common.ExecutionInstanceID():
                int(hal.Timer.Value.fpga_clock_cycles_per_us) * 100}] \
            * self.batch_size
        return inputs

    def test_extract_n_spikes(self):
        config = {grenade.common.ExecutionInstanceID():
                  lola.Chip.default_neuron_bypass}
        # Get graph
        network_graph, config = self.generate_network(config)
        # Get inputs
        inputs = self.generate_inputs(network_graph)
        # Get chip config
        data = _hxtorch_spiking.run(
            config, network_graph, inputs, {
                grenade.common.ExecutionInstanceID():
                grenade.signal_flow.ExecutionInstanceHooks()})
        spikes = _hxtorch_core.extract_n_spikes(
            data, network_graph,
            int(hal.Timer.Value.fpga_clock_cycles_per_us) * 100,
            {self.int_pop_descr: 2})

        self.assertEqual(len(spikes), 1)
        self.assertEqual(len(spikes[self.int_pop_descr]), 2)
        self.assertEqual(
                list(spikes[self.int_pop_descr][0].shape),
                [self.batch_size, 2])
        self.assertEqual(
                list(spikes[self.int_pop_descr][1].shape),
                [self.batch_size, 2])


if __name__ == "__main__":
    unittest.main()
