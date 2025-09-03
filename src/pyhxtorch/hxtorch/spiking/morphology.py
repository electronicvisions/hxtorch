'''
User defined neuron morphologies.
'''
from abc import ABC, abstractmethod
from typing import Tuple, Union
import pylogging as logger

import numpy as np

from dlens_vx_v3 import lola, hal, halco

log = logger.get("hxtorch.spiking.morphology")


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

    # pylint: disable=invalid-name
    @staticmethod
    def format_to_CapMemCell_value(**kwargs) -> Tuple[hal.CapMemCell.Value]:
        '''
        Helper function that can convert numbers (float or int) into
        hal.CapMemCell.Value type while issuing warnings, if bonds of the
        assignable value range are tried to surpass.

        :param kwargs: Dictionary that holds the values that are to be
            converted and their respective variable names (used for warnings)
        :returns: Tuple that holds the according hal.CapMemCell.Value for each
            passed value via kwargs
        '''

        return_values = []
        for param_name, param in kwargs.items():
            if param > 1022:
                log.WARN(f"Hardware value of model parameter {param_name} "
                         + f"({param}) exceeded maximal applicable value "
                         + "of 1022. Changing to 1022...")
                param = 1022
            if param < 0:
                log.WARN(f"Hardware value of model parameter {param_name} "
                         + f"({param}) undercuts minimal applicable value "
                         + "of 0. Changing to 0...")
                param = 0
            return_values.append(hal.CapMemCell.Value(int(param)))
        return tuple(return_values)

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

    @staticmethod
    def set_exponential_params(coord: halco.LogicalNeuronOnDLS,
                               neuron_block: lola.NeuronBlock,
                               exponential_threshold: Union[float, int],
                               exponential_slope: Union[float, int]) -> None:
        '''
        Set all parameters related to the exponential term of the adaptive
        exponential leaky integrate-and-fire model on the given hardware
        neuron.

        :param coord: Coordinate of the logical neuron for which the parameters
            are to be set.
        :param neuron_block: Neuron block in which the configuration of the
            atomic neurons is changed.
        :param exponential_threshold: Parameter value to be set for the
            exponential threshold.
        :param exponential_slope: Parameter value to be set for the
            exponential slope.
        '''

        config = neuron_block.atomic_neurons[coord.get_atomic_neurons()[0]]
        exponential_threshold, exponential_slope = \
            Morphology.format_to_CapMemCell_value(
                exponential_threshold=exponential_threshold,
                exponential_slope=exponential_slope)

        config.exponential.enable = True
        config.exponential.v_exp = exponential_threshold
        config.exponential.i_bias = exponential_slope

    @staticmethod
    def set_adaptation_base_params(coord: halco.LogicalNeuronOnDLS,
                                   neuron_block: lola.NeuronBlock,
                                   tau_adap: Union[float, int]) -> None:
        '''
        Set all parameters related to the base of the adaptation term of the
        adaptive exponential leaky integrate-and-fire model (without
        subthreshold- or spike-triggered adaptation) on the given hw neuron.

        :param coord: Coordinate of the logical neuron for which the parameters
            are to be set.
        :param neuron_block: Neuron block in which the configuration of the
            atomic neurons is changed.
        :param tau_adap: Parameter value to be set for the adaptation time
            constant.
        '''

        config = neuron_block.atomic_neurons[coord.get_atomic_neurons()[0]]
        tau_adap, = Morphology.format_to_CapMemCell_value(tau_adap=tau_adap)

        config.adaptation.enable = True
        config.adaptation.enable_pulse = False
        config.adaptation.v_ref = hal.CapMemCell.Value(int(511))
        config.adaptation.i_bias_tau = tau_adap

    # pylint: disable=invalid-name
    @staticmethod
    def set_subthreshold_adaptation_strength(
            coord: halco.LogicalNeuronOnDLS,
            neuron_block: lola.NeuronBlock,
            subthreshold_adaptation_strength: Union[float, int],
            leak_adaptation: Union[float, int, None]) -> None:
        '''
        Set the hardware parameter for the subthreshold adaptation strength
        on the given hw neuron.

        :param coord: Coordinate of the logical neuron for which the parameters
            are to be set.
        :param neuron_block: Neuron block in which the configuration of the
            atomic neurons is changed.
        :param subthreshold_adaptation_strength: Parameter value to be set for
            the subthreshold adaptation strength.
        :param leak_adaptation: Parameter value to be set for the leak
            potential from the membrane taken into account by the
            subthreshold adaptation mechanism on hardware.
        '''

        config = neuron_block.atomic_neurons[coord.get_atomic_neurons()[0]]
        a_is_negative = subthreshold_adaptation_strength < 0.
        subthreshold_adaptation_strength, = \
            Morphology.format_to_CapMemCell_value(
                subthreshold_adaptation_strength=abs(
                    subthreshold_adaptation_strength))
        leak_adaptation, = (config.leak.v_leak,) if leak_adaptation is None \
            else Morphology.format_to_CapMemCell_value(
                leak_adaptation=leak_adaptation)

        config.adaptation.i_bias_a = subthreshold_adaptation_strength
        config.adaptation.invert_a = a_is_negative
        config.adaptation.v_leak = leak_adaptation

    # pylint: disable=invalid-name
    @staticmethod
    def set_spike_triggered_adaptation_increment(
            coord: halco.LogicalNeuronOnDLS,
            neuron_block: lola.NeuronBlock,
            spike_triggered_adaptation_increment: Union[float, int],
            clock_scale_adaptation_pulse: Tuple[int] = (5, 5)) -> None:
        '''
        Set the hardware parameter for the spike-triggered adaptation increment
        on the given hw neuron.

        :param coord: Coordinate of the logical neuron for which the parameters
            are to be set.
        :param neuron_block: Neuron block in which the configuration of the
            atomic neurons is changed.
        :param spike_triggered_adaptation_increment: Parameter value to be set
            for the spike-triggered adaptation increment.
        '''

        config = neuron_block.atomic_neurons[coord.get_atomic_neurons()[0]]
        b_is_negative = spike_triggered_adaptation_increment < 0.
        spike_triggered_adaptation_increment, = \
            Morphology.format_to_CapMemCell_value(
                spike_triggered_adaptation_increment=abs(
                    spike_triggered_adaptation_increment))

        config.adaptation.enable_pulse = True
        neuron_block.backends[0].enable_clocks = True
        neuron_block.backends[0].clock_scale_adaptation_pulse = \
            clock_scale_adaptation_pulse[0]
        neuron_block.backends[1].enable_clocks = True
        neuron_block.backends[1].clock_scale_adaptation_pulse = \
            clock_scale_adaptation_pulse[1]
        config.adaptation.i_bias_b = spike_triggered_adaptation_increment
        config.adaptation.invert_b = b_is_negative


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
