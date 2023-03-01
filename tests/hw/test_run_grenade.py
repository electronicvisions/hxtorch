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
        network_builder = grenade.network.placed_atomic.NetworkBuilder()

        # Populations
        neurons = [
            halco.AtomicNeuronOnDLS(coord, halco.NeuronRowOnDLS.top)
            for coord in halco.iter_all(halco.NeuronColumnOnDLS)
            ][:self.int_pop_size]
        int_pop = grenade.network.placed_atomic.Population(
            neurons, [True] * len(neurons))
        ext_pop = grenade.network.placed_atomic.ExternalPopulation(
            self.ext_pop_size)
        int_pop_descr = network_builder.add(int_pop)
        self.ext_pop_descr = network_builder.add(ext_pop)

        # Some CADC recording
        cadc_recording = grenade.network.placed_atomic.CADCRecording()
        recorded_neurons = list()
        for nrn_id in range(self.int_pop_size):
            recorded_neurons.append(
                grenade.network.placed_atomic.CADCRecording.Neuron(
                    int_pop_descr, nrn_id,
                    lola.AtomicNeuron.Readout.Source.membrane))
        cadc_recording.neurons = recorded_neurons
        network_builder.add(cadc_recording)

        # Some connections
        connections = []
        for i in range(self.ext_pop_size):
            connections.append(
                grenade.network.placed_atomic.Projection.Connection(i, i, 63))
        proj = grenade.network.placed_atomic.Projection(
            grenade.network.placed_atomic.Projection.ReceptorType.excitatory,
            connections, self.ext_pop_descr, int_pop_descr)
        network_builder.add(proj)

        # Build network graph
        network = network_builder.done()
        routing_result = grenade.network.placed_atomic.build_routing(network)
        return grenade.network.placed_atomic.build_network_graph(
            network, routing_result)

    def generate_inputs(self, network_graph):
        # Inputs
        input_generator = grenade.network.placed_atomic.InputGenerator(
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
        inputs.runtime = {
            grenade.signal_flow.ExecutionInstance(): self.batch_size * [
                int(hal.Timer.Value.fpga_clock_cycles_per_us) * 100]}
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
