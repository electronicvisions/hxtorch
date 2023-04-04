'''
User defined neuron morphologies.
'''
from abc import ABC, abstractmethod

import numpy as np

from dlens_vx_v3 import lola, hal, halco


class Morphology(ABC):
    '''
    Represents the internal structure of a neuron.

    This neuron might be made up of several compartments and the compartments
    themselves can consist of several neuron circuits.

    :note: Currently spike and voltage recording is only supported in the
        first neuron circuit of the first compartment.
    '''
    @property
    @abstractmethod
    def compartments(self) -> halco.LogicalNeuronCompartments:
        '''
        Unplaced coordinate of the logical neuron.
        '''

    @property
    @abstractmethod
    def logical_neuron(self) -> lola.LogicalNeuron:
        '''
        Base configuration of the logical neuron.

        Default constructed logical neuron, the connections between neuron
        circuits are configured such that the specified morphology is
        implemented.
        '''

    def implement_morphology(self,
                             coord: halco.LogicalNeuronOnDLS,
                             neuron_block: lola.NeuronBlock) -> None:
        '''
        Configure the atomic neurons in the given neuron block to represent
        this morphology.

        :param coord: Coordinate of the logical neuron which should be
            configured.
        :param neuron_block: The configuration of neurons at `coord` will be
            changed such that a neuron with the given morphology is
            implemented.
        '''
        # collapse_neuron() converts MCSafeAtomicNeurons to AtomicNeurons
        ln_config = self.logical_neuron.collapse_neuron()
        # set morphology
        for comp, an_coords in coord.get_placed_compartments().items():
            for an_coord, an_config in zip(an_coords, ln_config[comp]):
                neuron_block.atomic_neurons[an_coord].multicompartment = \
                    an_config.multicompartment

    @staticmethod
    def enable_madc_recording(coord: halco.LogicalNeuronOnDLS,
                              neuron_block: lola.NeuronBlock,
                              readout_source: hal.NeuronConfig.ReadoutSource
                              ) -> None:
        '''
        Configure neuron such that traces can be recorded with the MADC.

        :param coord: Coordinate of the logical neuron for which the recording
            is enabled.
        :param neuron_block: Neuron block in which the configuration of the
            atomic neurons is changed.
        :param readout_source: Voltage which should be recorded.
        '''
        config = neuron_block.atomic_neurons[coord.get_atomic_neurons()[0]]
        config.readout.enable_amplifier = True
        config.readout.enable_buffered_access = True
        config.readout.source = readout_source

    @staticmethod
    def set_spike_recording(enable: bool,
                            coord: halco.LogicalNeuronOnDLS,
                            neuron_block: lola.NeuronBlock) -> None:
        '''
        Set whether spikes are forwarded digitally.

        :param enable: Enable/disable the digital routing of spikes.
        :param coord: Coordinate of the logical neuron for which the recording
            is enabled.
        :param neuron_block: Neuron block in which the configuration of the
            atomic neurons is changed.
        '''
        config = neuron_block.atomic_neurons[coord.get_atomic_neurons()[0]]
        config.event_routing.enable_digital = enable

    @staticmethod
    def disable_spiking(coord: halco.LogicalNeuronOnDLS,
                        neuron_block: lola.NeuronBlock) -> None:
        '''
        Disable spiking for the given neuron.

        Disable the threshold comparator and the digital spike output.

        :param coord: Coordinate of the logical neuron for which spiking is
            disabled.
        :param neuron_block: Neuron block in which the configuration of the
            atomic neurons is changed.
        '''

        for an_coord in coord.get_atomic_neurons():
            config = neuron_block.atomic_neurons[an_coord]
            config.threshold.enable = False
            config.event_routing.enable_digital = True

    @staticmethod
    def disable_leak(coord: halco.LogicalNeuronOnDLS,
                     neuron_block: lola.NeuronBlock) -> None:
        '''
        Disable the leak for the given neuron.

        :param coord: Coordinate of the logical neuron for which the leak is
            disabled.
        :param neuron_block: Neuron block in which the configuration of the
            atomic neurons is changed.
        '''

        for an_coord in coord.get_atomic_neurons():
            config = neuron_block.atomic_neurons[an_coord]
            config.leak.i_bias = 0
            config.leak.enable_division = True
            config.leak.enable_multiplication = False


class SingleCompartmentNeuron(Morphology):
    '''
    Neuron with a single iso-potential compartment.

    The compartment can consist of several neuron circuits. For all but the
    first neuron circuit leak, threshold and capacitance are disabled.
    '''

    def __init__(self, size: int, expand_horizontally: bool = False) -> None:
        '''
        Create a single-compartment neuron.

        :param size: Number of neuron circuits per compartment.
        :param expand_horizontally: Expand the neurons in the same row before
            starting a second row. If False, the columns are filled before
            the shape is expanded horizontally.
        '''
        super().__init__()

        if expand_horizontally:
            neurons = [
                halco.AtomicNeuronOnLogicalNeuron(halco.common.Enum(idx)) for
                idx in range(size)]
        else:
            all_neurons = np.array(
                list(halco.iter_all(halco.AtomicNeuronOnLogicalNeuron)))
            neurons = all_neurons.reshape(2, -1).ravel('F')[:size]

        morphology = lola.Morphology()
        morphology.create_compartment(neurons)
        self._compartments, self._logical_neuron = morphology.done()

    @property
    def compartments(self) -> halco.LogicalNeuronCompartments:
        return self._compartments

    @property
    def logical_neuron(self) -> lola.LogicalNeuron:
        return self._logical_neuron

    def implement_morphology(self,
                             coord: halco.LogicalNeuronOnDLS,
                             neuron_block: lola.NeuronBlock) -> None:
        assert len(coord.get_placed_compartments()) == 1
        assert len(self.compartments.get_compartments()) == 1

        super().implement_morphology(coord, neuron_block)

        self._one_active_circuit(coord, neuron_block)

    @staticmethod
    def _one_active_circuit(coord: halco.LogicalNeuronOnDLS,
                            neuron_block: lola.NeuronBlock) -> None:
        '''
        Enable fire signal forwarding for first circuit and disable leak,
        capacitance as well as threshold for all other circuit.
        '''
        for an_coord in coord.get_atomic_neurons()[1:]:
            config = neuron_block.atomic_neurons[an_coord]
            config.leak.i_bias = 0
            config.leak.enable_division = True
            config.leak.enable_multiplication = False
            config.membrane_capacitance.capacitance = 0
            config.threshold.enable = False

        config = neuron_block.atomic_neurons[coord.get_atomic_neurons()[0]]
        config.event_routing.analog_output = \
            config.EventRouting.AnalogOutputMode.normal
