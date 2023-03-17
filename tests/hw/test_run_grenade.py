"""
"""
import unittest
import torch
import hxtorch
import _hxtorch
import pygrenade_vx as grenade
from dlens_vx_v3 import halco, lola, hal


class TestRun(unittest.TestCase):

    batch_size: int = 3
    int_pop_size: int = 3
    ext_pop_size: int = 3

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware(calib_name="spiking")

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()

    def generate_network(self):
        # Builder
        network_builder = grenade.network.placed_logical.NetworkBuilder()

        # Populations
        neurons = [grenade.network.placed_logical.Population.Neuron(
            halco.LogicalNeuronOnDLS(halco.LogicalNeuronCompartments({
                halco.CompartmentOnLogicalNeuron():
                [halco.AtomicNeuronOnLogicalNeuron()]}),
                halco.AtomicNeuronOnDLS(coord, halco.NeuronRowOnDLS.top)),
            {halco.CompartmentOnLogicalNeuron(): grenade.network
             .placed_logical.Population.Neuron
             .Compartment(grenade.network.placed_logical.Population.Neuron
             .Compartment.SpikeMaster(0, True),
             [{grenade.network.placed_logical.Receptor(grenade.network
               .placed_logical.Receptor.ID(),
               grenade.network.placed_logical.Receptor.Type.excitatory)}])})
            for coord in halco.iter_all(halco.NeuronColumnOnDLS)][
                :self.int_pop_size]
        int_pop = grenade.network.placed_logical.Population(neurons)
        ext_pop = grenade.network.placed_logical.ExternalPopulation(
            self.ext_pop_size)
        int_pop_descr = network_builder.add(int_pop)
        self.ext_pop_descr = network_builder.add(ext_pop)

        # Some CADC recording
        cadc_recording = grenade.network.placed_logical.CADCRecording()
        recorded_neurons = list()
        for nrn_id in range(self.int_pop_size):
            recorded_neurons.append(
                grenade.network.placed_logical.CADCRecording.Neuron(
                    int_pop_descr, nrn_id,
                    halco.CompartmentOnLogicalNeuron(), 0,
                    lola.AtomicNeuron.Readout.Source.membrane))
        cadc_recording.neurons = recorded_neurons
        network_builder.add(cadc_recording)

        # Some connections
        connections = []
        for i in range(self.ext_pop_size):
            connections.append(
                grenade.network.placed_logical.Projection.Connection(
                    (i, halco.CompartmentOnLogicalNeuron()),
                    (i, halco.CompartmentOnLogicalNeuron()),
                    grenade.network.placed_logical.Projection.Connection
                    .Weight(63)))
        proj = grenade.network.placed_logical.Projection(
            grenade.network.placed_logical.Receptor(
                grenade.network.placed_logical.Receptor.ID(),
                grenade.network.placed_logical.Receptor.Type.excitatory),
            connections, self.ext_pop_descr, int_pop_descr)
        network_builder.add(proj)

        # Build network graph
        network = network_builder.done()
        routing_result = grenade.network.placed_logical.build_routing(network)
        return grenade.network.placed_logical.build_network_graph(
            network, routing_result)

    def generate_inputs(self, network_graph):
        # Inputs
        input_generator = grenade.network.placed_logical.InputGenerator(
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
            grenade.signal_flow.ExecutionInstance():
            int(hal.Timer.Value.fpga_clock_cycles_per_us) * 100}] \
            * self.batch_size
        return inputs

    def test_run(self):
        # Get graph
        network = self.generate_network()

        # Get inputs
        inputs = self.generate_inputs(network)

        # Get chip config
        config = hxtorch.get_chip()

        # Execute a couple times like you would when training a model
        for i in range(10):
            output = _hxtorch._snn.run(
                config, network, inputs,
                grenade.signal_flow.ExecutionInstancePlaybackHooks())
            print(output)


if __name__ == "__main__":
    unittest.main()
