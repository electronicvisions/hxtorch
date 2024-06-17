"""
Implementing integrate-and-fire neuron module
"""
# pylint: disable=too-many-lines
from __future__ import annotations
from typing import (
    TYPE_CHECKING, Callable, Dict, Type, Optional, List, NamedTuple, Union)
import pylogging as logger

import torch

from dlens_vx_v3 import lola, hal, halco

import hxtorch.spiking.functional as F
from hxtorch.spiking.handle import NeuronHandle
from hxtorch.spiking.morphology import Morphology
from hxtorch.spiking.modules.neuron import Neuron
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment
    from hxtorch.spiking.execution_instance import ExecutionInstance

log = logger.get("hxtorch.spiking.modules")


class IAFNeuron(Neuron):
    """
    Integrate-and-fire neuron
    Caveat:
    For execution on hardware, this module can only be used in conjunction with
    a preceding Synapse module.
    """

    output_type: Type = NeuronHandle

    # TODO: Integrate into API
    _madc_readout_source: hal.NeuronConfig.ReadoutSource = \
        hal.NeuronConfig.ReadoutSource.membrane
    _cadc_readout_source: lola.AtomicNeuron.Readout.Source \
        = lola.AtomicNeuron.Readout.Source.membrane

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(self, size: int, experiment: Experiment,
                 func: Union[Callable, torch.autograd.Function]
                 = F.cuba_iaf_integration,
                 execution_instance: Optional[ExecutionInstance] = None,
                 params: Union[NamedTuple, F.CUBAIAFParams]
                 = F.CUBAIAFParams(1. / 10e-6, 1. / 10e-6),
                 enable_spike_recording: bool = True,
                 enable_cadc_recording: bool = True,
                 enable_cadc_recording_placement_in_dram: bool = False,
                 enable_madc_recording: bool = False,
                 record_neuron_id: Optional[int] = None,
                 placement_constraint: Optional[
                     List[halco.LogicalNeuronOnDLS]] = None,
                 trace_offset: Union[Dict[halco.AtomicNeuronOnDLS, float],
                                     torch.Tensor, float] = 0.,
                 trace_scale: Union[Dict[halco.AtomicNeuronOnDLS, float],
                                    torch.Tensor, float] = 1.,
                 cadc_time_shift: int = 1, shift_cadc_to_first: bool = False,
                 interpolation_mode: str = "linear",
                 neuron_structure: Optional[Morphology] = None) -> None:
        """
        Initialize an IAFNeuron. This module creates a population of a non-
        leaking spiking neurons of size `size`. This module has a internal
        spiking mask, which allows to disable the event output and spike
        recordings of specific neurons within the layer. This is particularly
        useful for dropout.

        :param size: Size of the population.
        :param experiment: Experiment to append layer to.
        :param func: Callable function implementing the module's forward
            functionality or a torch.autograd.Function implementing the
            module's forward and backward operation. Defaults to `LIF`.
        :param execution_instance: Execution instance to place to.
        :param params: Neuron Parameters in case of mock neuron integration of
            for backward path. If func does have a param argument the params
            object will get injected automatically.
        :param enable_spike_recording: Boolean flag to enable or disable spike
            recording. Note, this does not disable the event out put of
            neurons. The event output has to be disabled via `mask`.
        :param enable_cadc_recording: Enables or disables parallel sampling of
            the populations membrane trace via the CADC. A maximum sample rate
            of 1.7us is possible.
        :param enable_cadc_recording_placement_in_dram: Whether to place CADC
            recording data into DRAM (period ~6us) or SRAM (period ~2us).
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
        :param neuron_structure: Structure of the neuron. If not supplied a
            single neuron circuit is used.
        """
        super().__init__(
            size, experiment, func, execution_instance,
            params, enable_spike_recording,
            enable_cadc_recording, enable_cadc_recording_placement_in_dram,
            enable_madc_recording, record_neuron_id,
            placement_constraint, trace_offset, trace_scale, cadc_time_shift,
            shift_cadc_to_first, interpolation_mode, neuron_structure)

    def configure_hw_entity(self, neuron_id: int,
                            neuron_block: lola.NeuronBlock,
                            coord: halco.LogicalNeuronOnDLS) \
            -> lola.NeuronBlock:
        """
        Disables the neurons leak to behave like a integrate-and-fire neuron.

        :param neuron_id: In-population neuron index.
        :param neuron_block: The neuron block hardware entity.
        :param coord: Coordinate of neuron on hardware.
        :returns: Configured neuron block.
        """
        neuron_block = super().configure_hw_entity(
            neuron_id, neuron_block, coord)
        self._neuron_structure.disable_leak(coord, neuron_block)
        return neuron_block
