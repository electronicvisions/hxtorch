"""
Test MNIST 16x16 dummy network construction and execution
"""
import unittest
from hxtorch.spiking.modules import neuron
from hxtorch.spiking.morphology import SingleCompartmentNeuron
import torch
import hxtorch
from hxtorch import spiking as hxsnn
import pygrenade_vx.network.placed_logical as grenade
from hxtorch.spiking.utils import calib_helper
from dlens_vx_v3 import hal, halco

hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.WARN)


class TestSNNCustomRouting256I246H10O(unittest.TestCase):
    """
    Test snn custom routing for 256 input, 246 hidden and 10 output units
    """

    @classmethod
    def setUpClass(cls):
        hxtorch.init_hardware()

    @classmethod
    def tearDownClass(cls):
        hxtorch.release_hardware()

    @staticmethod
    def hw_routing_func(network: grenade.Network) \
            -> grenade.RoutingResult:
        assert len(network.populations) == 3

        ret = grenade.RoutingResult()

        # set synapse row modes to signed double rows per synapse driver
        # excitatory even, inhibitory odd row index
        synapse_row_modes = {}
        for row in halco.iter_all(halco.SynapseRowOnDLS):
            synapse_row_modes[row] = hal.SynapseDriverConfig.RowMode\
                .inhibitory if row.toEnum().value() % 2 else hal\
                .SynapseDriverConfig.RowMode.excitatory
        ret.synapse_row_modes = synapse_row_modes

        # configure crossbar nodes
        crossbar_nodes = {}
        for coord in halco.iter_all(halco.CrossbarNodeOnDLS):
            crossbar_nodes[coord] = hal.CrossbarNode()

        # disable non-diagonal input from L2
        for cinput in halco.iter_all(halco.SPL1Address):
            for coutput in halco.iter_all(halco.PADIBusOnDLS):
                coord = halco.CrossbarNodeOnDLS(
                    coutput.toCrossbarOutputOnDLS(),
                    cinput.toCrossbarInputOnDLS())
                crossbar_nodes[coord] = hal.CrossbarNode.drop_all

        # enable input from L2 to top half
        for coutput in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            coord = halco.CrossbarNodeOnDLS(
                halco.PADIBusOnDLS(coutput, halco.PADIBusBlockOnDLS.top)
                .toCrossbarOutputOnDLS(),
                halco.SPL1Address(coutput).toCrossbarInputOnDLS())
            config = hal.CrossbarNode()
            config.mask = halco.NeuronLabel(1 << 13)
            config.target = halco.NeuronLabel(0 << 13)
            crossbar_nodes[coord] = config

        # enable input from L2 to bottom half
        for coutput in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            coord = halco.CrossbarNodeOnDLS(
                halco.PADIBusOnDLS(coutput, halco.PADIBusBlockOnDLS.bottom)
                .toCrossbarOutputOnDLS(),
                halco.SPL1Address(coutput).toCrossbarInputOnDLS())
            config = hal.CrossbarNode()
            config.mask = halco.NeuronLabel(1 << 13)
            config.target = halco.NeuronLabel(1 << 13)
            crossbar_nodes[coord] = config

        # enable input from left half to top half
        for output in halco.iter_all(
                halco.NeuronEventOutputOnNeuronBackendBlock):
            coord = halco.CrossbarNodeOnDLS(
                halco.PADIBusOnDLS(
                    halco.PADIBusOnPADIBusBlock(output),
                    halco.PADIBusBlockOnDLS.top).toCrossbarOutputOnDLS(),
                halco.NeuronEventOutputOnDLS(
                    output, halco.NeuronBackendConfigBlockOnDLS(0))
                .toCrossbarInputOnDLS())
            config = hal.CrossbarNode()
            config.mask = halco.NeuronLabel(1 << 13)
            config.target = halco.NeuronLabel(1 << 13)
            crossbar_nodes[coord] = config

        # enable input from right half to bottom half
        for output in halco.iter_all(
                halco.NeuronEventOutputOnNeuronBackendBlock):
            coord = halco.CrossbarNodeOnDLS(
                halco.PADIBusOnDLS(
                    halco.PADIBusOnPADIBusBlock(output),
                    halco.PADIBusBlockOnDLS.bottom).toCrossbarOutputOnDLS(),
                halco.NeuronEventOutputOnDLS(
                    output, halco.NeuronBackendConfigBlockOnDLS(1))
                .toCrossbarInputOnDLS())
            config = hal.CrossbarNode()
            config.mask = halco.NeuronLabel(1 << 13)
            config.target = halco.NeuronLabel(1 << 13)
            crossbar_nodes[coord] = config

        ret.crossbar_nodes = crossbar_nodes

        # no separation via synapse driver compare masks
        synapse_driver_compare_masks = {}
        for driver in halco.iter_all(halco.SynapseDriverOnDLS):
            synapse_driver_compare_masks[driver] = 0
        synapse_driver_compare_masks[
            halco.SynapseDriverOnDLS(
                halco.SynapseDriverOnSynapseDriverBlock(123),
                halco.SynapseDriverBlockOnDLS(1))] = 0b00010
        synapse_driver_compare_masks[
            halco.SynapseDriverOnDLS(
                halco.SynapseDriverOnSynapseDriverBlock(124),
                halco.SynapseDriverBlockOnDLS(1))] = 0b00010
        synapse_driver_compare_masks[
            halco.SynapseDriverOnDLS(
                halco.SynapseDriverOnSynapseDriverBlock(125),
                halco.SynapseDriverBlockOnDLS(1))] = 0b00010
        synapse_driver_compare_masks[
            halco.SynapseDriverOnDLS(
                halco.SynapseDriverOnSynapseDriverBlock(126),
                halco.SynapseDriverBlockOnDLS(1))] = 0b00010
        synapse_driver_compare_masks[
            halco.SynapseDriverOnDLS(
                halco.SynapseDriverOnSynapseDriverBlock(127),
                halco.SynapseDriverBlockOnDLS(1))] = 0b00010
        ret.synapse_driver_compare_masks = synapse_driver_compare_masks

        #print(network)

        # use internal neuron labels linearly
        neurons_0 = network.populations[
            grenade.PopulationDescriptor(0)].neurons
        assert len(neurons_0) == 246
        internal_neuron_labels_0 = []
        for i in range(246):
            internal_neuron_labels_0.append({halco.CompartmentOnLogicalNeuron(): [(i & 0b00011111) | 0b00000000, None]})
        ret.internal_neuron_labels[grenade.PopulationDescriptor(0)] = \
            internal_neuron_labels_0
        neurons_1 = network.populations[
            grenade.PopulationDescriptor(1)].neurons
        assert len(neurons_1) == 10
        internal_neuron_labels_1 = []
        for i in range(246, 256):
            internal_neuron_labels_1.append({halco.CompartmentOnLogicalNeuron(): [(i & 0b00011111) | 0b00000000, None]})
        ret.internal_neuron_labels = {
            grenade.PopulationDescriptor(0):
            internal_neuron_labels_0,
            grenade.PopulationDescriptor(1):
            internal_neuron_labels_1
        }

        # linearly assign input event label to synapse driver
        external_spike_labels = []
        input_size = network.populations[
            grenade.PopulationDescriptor(2)].size
        for i in range(input_size):
            label = halco.SpikeLabel(
                ((i < input_size // 2) << 13)  # top/bottom hemisphere
                | (((i // halco.SynapseDriverOnPADIBus.size)
                    % halco.PADIBusOnPADIBusBlock.size) << 14)  # PADI-bus selection
                | (0b00010 << 6)  # deselection of last 10 drivers for hidden -> output layer
                | ((i % halco.SynapseDriverOnPADIBus.size) + 32)  # unused synapse label above internal neuron labels
            )
            external_spike_labels.append([label])
        ret.external_spike_labels = {
            grenade.PopulationDescriptor(2):
            external_spike_labels}

        # linearly place projections
        connections = {}
        connection_routing_result = {}
        for j in range(0, 2):
            projection = network.projections[
                grenade.ProjectionDescriptor(j)]
            is_inh = projection.receptor.type == grenade.Receptor.Type\
                .inhibitory
            connections_ho = []
            for i, connection in enumerate(network.projections[
                    grenade.ProjectionDescriptor(j)].connections):
                placed_connection = grenade.RoutingResult\
                    .PlacedConnection()
                placed_connection.weight = connection.weight.value()
                placed_connection.synapse_on_row = network.populations[
                    projection.population_post].neurons[connection.index_post[0]]\
                    .coordinate.get_placed_compartments()[connection.index_post[1]][0].toNeuronColumnOnDLS().toSynapseOnSynapseRow()
                placed_connection.synapse_row = \
                    halco.SynapseRowOnDLS(
                        halco.common.Enum(2 * connection.index_pre[0] + is_inh))
                placed_connection.label = (connection.index_pre[0] % 32) + 32
                connections_ho.append([placed_connection])
                routes = grenade.ConnectionToHardwareRoutes()
                routes.atomic_neurons_on_target_compartment = [placed_connection.synapse_row.toSynramOnDLS().value()]
                if grenade.ProjectionDescriptor(j) not in connection_routing_result:
                    connection_routing_result.update({
                        grenade.ProjectionDescriptor(j): []})
                connection_routing_result[grenade.ProjectionDescriptor(j)].append(routes)
            connections[
                grenade.ProjectionDescriptor(j)] = connections_ho

        for j in range(2, 4):
            projection = network.projections[
                grenade.ProjectionDescriptor(j)]
            is_inh = projection.receptor.type == grenade.Receptor.Type\
                .inhibitory
            connections_ih = []
            for i, connection in enumerate(network.projections[
                    grenade.ProjectionDescriptor(j)].connections):
                placed_connection = grenade.RoutingResult\
                    .PlacedConnection()
                placed_connection.weight = connection.weight.value()
                placed_connection.synapse_on_row = network.populations[
                    projection.population_post].neurons[connection.index_post[0]]\
                    .coordinate.get_placed_compartments()[connection.index_post[1]][0].toNeuronColumnOnDLS().toSynapseOnSynapseRow()
                placed_connection.synapse_row = halco.SynapseRowOnDLS(
                    halco.common.Enum(2 * connection.index_pre[0] + is_inh))
                placed_connection.label = (connection.index_pre[0] // 2) % 32
                connections_ih.append([placed_connection])
                routes = grenade.ConnectionToHardwareRoutes()
                routes.atomic_neurons_on_target_compartment = [placed_connection.synapse_row.toSynramOnDLS().value()]
                if grenade.ProjectionDescriptor(j) not in connection_routing_result:
                    connection_routing_result.update({
                        grenade.ProjectionDescriptor(j): []})
                connection_routing_result[grenade.ProjectionDescriptor(j)].append(routes)
            connections[grenade.ProjectionDescriptor(j)] = connections_ih
        ret.connections = connections
        ret.connection_routing_result = connection_routing_result

        return ret

    def test(self):
        experiment = hxsnn.Experiment(hw_routing_func=self.hw_routing_func)
        experiment.load_calib(calib_helper.nightly_calib_path())
        synapse_ih = hxsnn.Synapse(256, 246, experiment=experiment)
        synapse_ih.weight.data = (torch.rand(synapse_ih.weight.shape) - 0.5) * 126.
        neuron_h = hxsnn.Neuron(
            246, experiment, neuron_structure=SingleCompartmentNeuron(2))
        synapse_ho = hxsnn.Synapse(246, 10, experiment=experiment)
        synapse_ho.weight.data = (torch.rand(synapse_ho.weight.shape) - 0.5) * 126.
        neuron_o = hxsnn.ReadoutNeuron(
            10, experiment, neuron_structure=SingleCompartmentNeuron(2))
        # Test output handle
        input = hxsnn.NeuronHandle(
            spikes=torch.bernoulli(torch.ones((10, 10, 256)) * 0.5))

        synapse_ih_handle = synapse_ih(input)
        neuron_h_handle = neuron_h(synapse_ih_handle)
        synapse_ho_handle = synapse_ho(neuron_h_handle)
        neuron_o_handle = neuron_o(synapse_ho_handle)

        hxsnn.run(experiment, 10)

        # Assert spikes exist
        self.assertIsInstance(neuron_o_handle.v_cadc, torch.Tensor)


if __name__ == "__main__":
    unittest.main()