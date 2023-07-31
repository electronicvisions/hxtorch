"""
"""
import unittest
import hxtorch
import _hxtorch_spiking
import pygrenade_vx as grenade
from dlens_vx_v3 import halco, lola, hal, sta


class TestRun(unittest.TestCase):

    batch_size: int = 3
    int_pop_size: int = 3
    ext_pop_size: int = 3

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()

    def generate_network(self):
        # Builder
        network_builder = grenade.network.NetworkBuilder()

        # Populations
        neurons = [grenade.network.Population.Neuron(
            halco.LogicalNeuronOnDLS(halco.LogicalNeuronCompartments({
                halco.CompartmentOnLogicalNeuron():
                [halco.AtomicNeuronOnLogicalNeuron()]}),
                halco.AtomicNeuronOnDLS(coord, halco.NeuronRowOnDLS.top)),
            {halco.CompartmentOnLogicalNeuron(): grenade.network
             .Population.Neuron
             .Compartment(grenade.network.Population.Neuron
             .Compartment.SpikeMaster(0, True),
             [{grenade.network.Receptor(grenade.network
               .Receptor.ID(),
               grenade.network.Receptor.Type.excitatory)}])})
            for coord in halco.iter_all(halco.NeuronColumnOnDLS)][
                :self.int_pop_size]
        int_pop = grenade.network.Population(neurons)
        ext_pop = grenade.network.ExternalSourcePopulation(
            self.ext_pop_size)
        int_pop_descr = network_builder.add(int_pop)
        self.ext_pop_descr = network_builder.add(ext_pop)

        # Some CADC recording
        cadc_recording = grenade.network.CADCRecording()
        recorded_neurons = list()
        for nrn_id in range(self.int_pop_size):
            recorded_neurons.append(
                grenade.network.CADCRecording.Neuron(
                    int_pop_descr, nrn_id,
                    halco.CompartmentOnLogicalNeuron(), 0,
                    lola.AtomicNeuron.Readout.Source.membrane))
        cadc_recording.neurons = recorded_neurons
        network_builder.add(cadc_recording)

        # Some connections
        connections = []
        for i in range(self.ext_pop_size):
            connections.append(
                grenade.network.Projection.Connection(
                    (i, halco.CompartmentOnLogicalNeuron()),
                    (i, halco.CompartmentOnLogicalNeuron()),
                    grenade.network.Projection.Connection
                    .Weight(63)))
        proj = grenade.network.Projection(
            grenade.network.Receptor(
                grenade.network.Receptor.ID(),
                grenade.network.Receptor.Type.excitatory),
            connections, self.ext_pop_descr, int_pop_descr)
        network_builder.add(proj)

        # Build network graph
        network = network_builder.done()
        routing_result = grenade.network.routing\
            .PortfolioRouter()(network)
        return grenade.network.build_network_graph(
            network, routing_result)

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
        identifier = hxtorch.get_unique_identifier()
        path = f"/wang/data/calibration/hicann-dls-sr-hx/{identifier}/stable/"\
            + "latest/spiking_cocolist.pbin"
        with open(path, 'rb') as fd:
            data = fd.read()
        dumper = sta.DumperDone()
        sta.from_portablebinary(dumper, data)
        config = sta.convert_to_chip(dumper)

        # Execute a couple times like you would when training a model
        for i in range(10):
            output = _hxtorch_spiking.run(
                config, network, inputs,
                grenade.signal_flow.ExecutionInstancePlaybackHooks())
            print(output)


if __name__ == "__main__":
    unittest.main()
