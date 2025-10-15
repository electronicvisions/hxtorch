'''
User defined neuron morphologies.
'''
from abc import ABC, abstractmethod
from typing import (
    Dict,
    List,
    Tuple,
    Union,
)
import pylogging as logger

import numpy as np

from dlens_vx_v3 import lola, hal, halco
import pygrenade_vx as grenade

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

    def implement_morphology(
        self,
        coord: halco.LogicalNeuronOnDLS,
        configs: Dict[grenade.common.CompartmentOnNeuron,
                      List[halco.AtomicNeuronOnDLS]],
    ) -> None:
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
        for comp, _ in coord.get_placed_compartments().items():
            gcomp = grenade.common.CompartmentOnNeuron(comp)
            for config, an_config in zip(configs[gcomp], ln_config[comp]):
                config.multicompartment = an_config.multicompartment

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
    def enable_madc_recording(
        configs: Dict[grenade.common.CompartmentOnNeuron,
                      List[halco.AtomicNeuronOnDLS]],
        readout_source: hal.NeuronConfig.ReadoutSource,
    ) -> None:
        '''
        Configure neuron such that traces can be recorded with the MADC.

        :param coord: Coordinate of the logical neuron for which the recording
            is enabled.
        :param neuron_block: Neuron block in which the configuration of the
            atomic neurons is changed.
        :param readout_source: Voltage which should be recorded.
        '''
        comp = grenade.common.CompartmentOnNeuron()
        configs[comp][0].readout.enable_amplifier = True
        configs[comp][0].readout.enable_buffered_access = True
        configs[comp][0].readout.source = readout_source

    @staticmethod
    def set_spike_recording(
        enable: bool,
        configs: Dict[grenade.common.CompartmentOnNeuron,
                      List[halco.AtomicNeuronOnDLS]],
    ) -> None:
        '''
        Set whether spikes are forwarded digitally.

        :param enable: Enable/disable the digital routing of spikes.
        :param coord: Coordinate of the logical neuron for which the recording
            is enabled.
        :param neuron_block: Neuron block in which the configuration of the
            atomic neurons is changed.
        '''
        comp = grenade.common.CompartmentOnNeuron()
        configs[comp][0].event_routing.enable_digital = enable

    @staticmethod
    def disable_spiking(configs: Dict[grenade.common.CompartmentOnNeuron,
                                      List[halco.AtomicNeuronOnDLS]]) -> None:
        '''
        Disable spiking for the given neuron.

        Disable the threshold comparator and the digital spike output.

        :param coord: Coordinate of the logical neuron for which spiking is
            disabled.
        :param neuron_block: Neuron block in which the configuration of the
            atomic neurons is changed.
        '''
        comp = grenade.common.CompartmentOnNeuron()
        for config in configs[comp]:
            config.threshold.enable = False
            config.event_routing.enable_digital = True

    @staticmethod
    def disable_leak(
        configs: Dict[grenade.common.CompartmentOnNeuron,
                      List[halco.AtomicNeuronOnDLS]],
    ) -> None:
        '''
        Disable the leak for the given neuron.

        :param coord: Coordinate of the logical neuron for which the leak is
            disabled.
        :param neuron_block: Neuron block in which the configuration of the
            atomic neurons is changed.
        '''
        comp = grenade.common.CompartmentOnNeuron()
        for config in configs[comp]:
            config.leak.i_bias = 0
            config.leak.enable_division = True
            config.leak.enable_multiplication = False

    @staticmethod
    def set_exponential_params(
        configs: Dict[grenade.common.CompartmentOnNeuron,
                      List[halco.AtomicNeuronOnDLS]],
        exponential_threshold: Union[float, int],
        exponential_slope: Union[float, int],
    ) -> None:
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
        # TODO: Enable for all atomic neurons?
        comp = grenade.common.CompartmentOnNeuron()
        for config in configs[comp]:
            exponential_threshold, exponential_slope = \
                Morphology.format_to_CapMemCell_value(
                    exponential_threshold=exponential_threshold,
                    exponential_slope=exponential_slope,
                )
            config.exponential.enable = True
            config.exponential.v_exp = exponential_threshold
            config.exponential.i_bias = exponential_slope

    @staticmethod
    def set_adaptation_base_params(
        configs: Dict[grenade.common.CompartmentOnNeuron,
                      List[halco.AtomicNeuronOnDLS]],
        tau_adap: Union[float, int],
    ) -> None:
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
        # TODO: Enable for all atomic neurons?
        comp = grenade.common.CompartmentOnNeuron()
        for config in configs[comp]:
            tau_adap, = Morphology.format_to_CapMemCell_value(
                tau_adap=tau_adap,
            )
            config.adaptation.enable = True
            config.adaptation.enable_pulse = False
            config.adaptation.v_ref = hal.CapMemCell.Value(int(511))
            config.adaptation.i_bias_tau = tau_adap

    # pylint: disable=invalid-name
    @staticmethod
    def set_subthreshold_adaptation_strength(
        configs: Dict[grenade.common.CompartmentOnNeuron,
                      List[halco.AtomicNeuronOnDLS]],
        subthreshold_adaptation_strength: Union[float, int],
        leak_adaptation: Union[float, int, None],
    ) -> None:
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
        # TODO: Enable for all atomic neurons?
        comp = grenade.common.CompartmentOnNeuron()
        for config in configs[comp]:
            a_is_negative = subthreshold_adaptation_strength < 0.
            subthreshold_adaptation_strength, = \
                Morphology.format_to_CapMemCell_value(
                    subthreshold_adaptation_strength=abs(
                        subthreshold_adaptation_strength))
            if leak_adaptation is None:
                leak_adaptation, = (config.leak.v_leak,)
            else:
                leak_adaptation, = Morphology.format_to_CapMemCell_value(
                    leak_adaptation=leak_adaptation)

            config.adaptation.i_bias_a = subthreshold_adaptation_strength
            config.adaptation.invert_a = a_is_negative
            config.adaptation.v_leak = leak_adaptation

    # pylint: disable=invalid-name
    @staticmethod
    def set_spike_triggered_adaptation_increment(
        configs: Dict[grenade.common.CompartmentOnNeuron,
                      List[halco.AtomicNeuronOnDLS]],
        backends: List[hal.CommonNeuronBackendConfig],
        spike_triggered_adaptation_increment: Union[float, int],
        clock_scale_adaptation_pulse: Tuple[int] = (5, 5),
    ) -> None:
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
        comp = grenade.common.CompartmentOnNeuron()
        for config in configs[comp]:
            b_is_negative = spike_triggered_adaptation_increment < 0.
            spike_triggered_adaptation_increment, = \
                Morphology.format_to_CapMemCell_value(
                    spike_triggered_adaptation_increment=abs(
                        spike_triggered_adaptation_increment))

            config.adaptation.enable_pulse = True
            config.adaptation.i_bias_b = spike_triggered_adaptation_increment
            config.adaptation.invert_b = b_is_negative

        backends[0].enable_clocks = True
        backends[0].clock_scale_adaptation_pulse = \
            clock_scale_adaptation_pulse[0]
        backends[1].enable_clocks = True
        backends[1].clock_scale_adaptation_pulse = \
            clock_scale_adaptation_pulse[1]

    @staticmethod
    def enable_constant_current(
        configs: Dict[grenade.common.CompartmentOnNeuron,
                      List[halco.AtomicNeuronOnDLS]],
        current_type: lola.AtomicNeuron.ConstantCurrent.Type
            = lola.AtomicNeuron.ConstantCurrent.Type.source
    ) -> None:
        '''
        Enable constant current input for the given neuron.

        :param coord: Coordinate of the logical neuron for which constant
            current input is enabled.
        :param neuron_block: Neuron block in which the configuration of the
            atomic neurons is changed.
        '''
        comp = grenade.common.CompartmentOnNeuron()
        configs[comp][0].constant_current.i_offset = 1000
        configs[comp][0].constant_current.enable = True
        configs[comp][0].constant_current.type = current_type


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

    def implement_morphology(
        self,
        coord,
        configs: Dict[grenade.common.CompartmentOnNeuron,
                      List[halco.AtomicNeuronOnDLS]],
    ) -> None:
        super().implement_morphology(coord, configs)
        self._one_active_circuit(configs)

    @staticmethod
    def _one_active_circuit(
        configs: Dict[grenade.common.CompartmentOnNeuron,
                      List[halco.AtomicNeuronOnDLS]],
    ) -> None:
        '''
        Enable fire signal forwarding for first circuit and disable leak,
        capacitance as well as threshold for all other circuit.
        '''
        comp = grenade.common.CompartmentOnNeuron()
        for config in configs[comp][1:]:
            config.leak.i_bias = 0
            config.leak.enable_division = True
            config.leak.enable_multiplication = False
            config.membrane_capacitance.capacitance = 0
            config.threshold.enable = False

        config = configs[comp][0]
        configs[comp][0].event_routing.analog_output = \
            config.EventRouting.AnalogOutputMode.normal
