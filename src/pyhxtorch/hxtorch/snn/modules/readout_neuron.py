"""
Implementing SNN modules
"""
from typing import (
    Callable, Dict, Tuple, Type, Optional, List, NamedTuple, Union)

import torch

from dlens_vx_v3 import lola, halco

from _hxtorch._snn import SpikeHandle, CADCHandle, MADCHandle  # pylint: disable=import-error
import hxtorch
import hxtorch.snn.functional as F
from hxtorch.snn.handle import ReadoutNeuronHandle
from hxtorch.snn.modules.neuron import Neuron

log = hxtorch.logger.get("hxtorch.snn.modules")


class ReadoutNeuron(Neuron):
    """
    Readout neuron layer

    Caveat:
    For execution on hardware, this module can only be used in conjuction with
    a preceding Synapse module.
    """

    output_type: Type = ReadoutNeuronHandle

    # pylint: disable=too-many-arguments
    def __init__(self, size: int, instance: "Instance",
                 func: Union[Callable, torch.autograd.Function] = F.LI,
                 params: Optional[NamedTuple] = None,
                 enable_cadc_recording: bool = True,
                 enable_madc_recording: bool = False,
                 record_neuron_id: Optional[int] = None,
                 placement_constraint: Optional[
                     List[halco.LogicalNeuronOnDLS]] = None,
                 trace_offset: Union[Dict[halco.AtomicNeuronOnDLS, float],
                                     torch.Tensor, float] = 0.,
                 trace_scale: Union[Dict[halco.AtomicNeuronOnDLS, float],
                                    torch.Tensor, float] = 1.,
                 cadc_time_shift: int = 1, shift_cadc_to_first: bool = False,
                 interpolation_mode: str = "linear") -> None:
        """
        Initialize a ReadoutNeuron. This module creates a population of non-
        spiking neurons of size `size` and is equivalent to Neuron when its
        spiking mask is disabled for all neurons.

        :param size: Size of the population.
        :param instance: Instance to append layer to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation. Defaults to `LI`.
        :param params: Neuron Parameters in case of mock neuron integration of
            for backward path. If func does have a param argument the params
            object will get injected automatically.
        :param enable_cadc_recording: Enables or disables parallel sampling of
            the populations membrane trace via the CADC. A maximum sample rate
            of 1.7us is possible.
        :param enable_madc_recording: Enables or disables the recording of the
            neurons `record_neuron_id` membrane trace via the MADC. Only a
            single neuron can be recorded. This membrane traces is samples with
            a significant higher resolution as with the CADC.
        :param record_neuron_id: The in-population neuron index of the neuron
            to be recorded with the MADC. This has only an effect when
            `enable_madc_recording` is enabled.
        :param placement_constraint: An optional list of logical neurons
            defining where to place the module`s neurons on hardware.
        :param trace_offset: The value by which the measured CADC traces are
            shifted before the scaling is applied. If this offset is given as
            float the same value is applied to all neuron traces in this
            population. One can also provide a torch tensor holding one offset
            for each individual neuron in this population. The corresponding
            tensor has to be of size `size`. Further, the offsets can be
            supplied in a dictionary where the keys are the hardware neuron
            coordinates and the values are the offsets, i.e.
            Dict[AtomicNeuronOnDLS, float]. The dictionary has to provide one
            coordinate for each hardware neuron represented by this population,
            but might also hold neuron coordinates that do not correspond to
            this layer. The layer-specific offsets are then picked and applied
            implicitly.
        :param trace_scale: The value by which the measured CADC traces are
            scaled after the offset is applied. If this scale is given as
            float all neuron traces are scaled with the same value population.
            One can also provide a torch tensor holding one scale for each
            individual neuron in this population. The corresponding tensor has
            to be of size `size`. Further, the scales can be supplied in a
            dictionary where the keys are the hardware neuron coordinates and
            the values are the scales, i.e. Dict[AtomicNeuronOnDLS, float]. The
            dictionary has to provide one coordinate for each hardware neuron
            represented by this population, but might also hold neuron
            coordinates that do not correspond to this layer. The layer-
            specific scales are then picked and applied implicitly.
        :param cadc_time_shift: An integer indicating by how many time steps
            the CADC values are shifted in time. A positive value shifts later
            CADC samples to earlier times and vice versa for a negative value.
        :param shift_cadc_to_first: A boolean indicating that the first
            measured CADC value is used as an offset. Note, this disables the
            param `trace_offset`.
        :param interpolation_mode: The method used to interpolate the measured
            CADC traces onto the given time grid.
        """
        super().__init__(
            size, instance, func, params, False, enable_cadc_recording,
            enable_madc_recording, record_neuron_id, placement_constraint,
            trace_offset, trace_scale, cadc_time_shift, shift_cadc_to_first,
            interpolation_mode)

    def configure_hw_entity(self, neuron_id: int,
                            neuron_block: lola.NeuronBlock,
                            coord: halco.LogicalNeuronOnDLS) \
            -> lola.NeuronBlock:
        """
        Configures a neuron in the given module with its specific properties.
        The neurons digital event outputs are enabled according to the given
        spiking mask.

        TODO: Additional parameterization should happen here, i.e. with
              population-specific parameters.

        :param neuron_id: In-population neuron index.
        :param neuron_block: The neuron block hardware entity.
        :param coord: Coordinate of neuron on hardware.

        :returns: Returns the AtomicNeuron with population-specific
            configurations appended.
        """
        atomic_neuron = neuron_block.atomic_neurons[
            coord.get_placed_compartments()[
                halco.CompartmentOnLogicalNeuron(0)][0]]

        # configure spike recording
        atomic_neuron.event_routing.analog_output = \
            atomic_neuron.EventRouting.AnalogOutputMode.normal
        atomic_neuron.event_routing.enable_digital = False

        # disable threshold comparator
        atomic_neuron.threshold.enable = False

        neuron_block.atomic_neurons[
            coord.get_placed_compartments()[
                halco.CompartmentOnLogicalNeuron(0)][0]] = atomic_neuron
        return neuron_block

    def post_process(self, hw_spikes: Optional[SpikeHandle],
                     hw_cadc: Optional[CADCHandle],
                     hw_madc: Optional[MADCHandle]) \
            -> Tuple[Optional[torch.Tensor], ...]:
        """
        User defined post process method called as soon as population-specific
        hardware observables are returned. This function has to convert the
        data types returned by grenade into PyTorch tensors. This function can
        be overridden by the user if non-default grenade-PyTorch data type
        conversion is required.
        Note: This function should return Tuple[Optional[torch.Tensor], ...],
              like (cadc or madc,). This should match the
              ReadoutTensorHandle signature.

        :param hw_spikes: A SpikeHandle holding the population's spikes
            recorded by grenade as a sparse tensor. This data can be ignored
            for this readout neuron.
        :param hw_cadc: The CADCHandle holding the CADC membrane readout
            events in a sparse tensor.
        :param hw_madc: The MADCHandle holding the MADC membrane readout
            events in a sparse tensor.

        :returns: Returns a tuple of optional torch.Tensors holding the
            hardware data (madc or cadc,)
        """
        # No spikes here
        assert not self._enable_spike_recording
        _, cadc, madc = super().post_process(hw_spikes, hw_cadc, hw_madc)

        return cadc, madc
