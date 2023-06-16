"""
Implementing SNN modules
"""
from __future__ import annotations
from typing import (
    TYPE_CHECKING, Callable, Dict, Tuple, Type, Optional, NamedTuple, Union,
    List)
import pylogging as logger
import numpy as np

import torch

from dlens_vx_v3 import lola, hal, halco
import pygrenade_vx.network.placed_logical as grenade

from _hxtorch_spiking import SpikeHandle, CADCHandle, MADCHandle  # pylint: disable=import-error
import hxtorch.spiking.functional as F
from hxtorch.spiking.morphology import Morphology, SingleCompartmentNeuron
from hxtorch.spiking.handle import NeuronHandle
from hxtorch.spiking.modules.hx_module import HXModule
from hxtorch.spiking.modules.types import Population
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment

log = logger.get("hxtorch.spiking.modules")


class Neuron(Population):
    """
    Neuron layer

    Caveat:
    For execution on hardware, this module can only be used in conjuction with
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
                 func: Union[Callable, torch.autograd.Function] = F.LIF,
                 params: Optional[NamedTuple] = None,
                 enable_spike_recording: bool = True,
                 enable_cadc_recording: bool = True,
                 enable_madc_recording: bool = False,
                 record_neuron_id: Optional[int] = None,
                 placement_constraint: Optional[
                     List[halco.LogicalNeuronOnDLS]] = None,
                 trace_offset: Union[Dict[halco.LogicalNeuronOnDLS, float],
                                     torch.Tensor, float] = 0.,
                 trace_scale: Union[Dict[halco.LogicalNeuronOnDLS, float],
                                    torch.Tensor, float] = 1.,
                 cadc_time_shift: int = 1, shift_cadc_to_first: bool = False,
                 interpolation_mode: str = "linear",
                 neuron_structure: Optional[Morphology] = None) -> None:
        """
        Initialize a Neuron. This module creates a population of spiking
        neurons of size `size`. This module has a internal spiking mask, which
        allows to disable the event ouput and spike recordings of specific
        neurons within the layer. This is particularly useful for dropout.

        :param size: Size of the population.
        :param experiment: Experiment to append layer to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation. Defaults to `LIF`.
        :param params: Neuron Parameters in case of mock neuron integration of
            for backward path. If func does have a param argument the params
            object will get injected automatically.
        :param enable_spike_recording: Boolean flag to enable or disable spike
            recording. Note, this does not disable the event out put of
            neurons. The event output has to be disabled via `mask`.
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
            supplied in a dictionary where the keys are the logical neuron
            coordinates and the values are the offsets, i.e.
            Dict[LogicalNeuronOnDLS, float]. The dictionary has to provide one
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
            dictionary where the keys are the logical neuron coordinates and
            the values are the scales, i.e. Dict[LogicalNeuronOnDLS, float].
            The dictionary has to provide one coordinate for each hardware
            neuron represented by this population, but might also hold neuron
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
        super().__init__(experiment=experiment, func=func)

        if placement_constraint is not None \
                and len(placement_constraint) != size:
            raise ValueError(
                "The number of neurons in logical neurons in "
                + "`hardware_constraints` does not equal the `size` of the "
                + "module.")

        self.size = size
        self.params = params
        self.extra_kwargs.update({"params": params, "dt": experiment.dt})

        self._enable_spike_recording = enable_spike_recording
        self._enable_cadc_recording = enable_cadc_recording
        self._enable_madc_recording = enable_madc_recording
        self._record_neuron_id = record_neuron_id
        self._placement_constraint = placement_constraint
        self._mask: Optional[torch.Tensor] = None
        self.unit_ids: Optional[np.ndarray] = None

        self.scale = trace_scale
        self.offset = trace_offset
        self.cadc_time_shift = cadc_time_shift
        self.shift_cadc_to_first = shift_cadc_to_first

        self.interpolation_mode = interpolation_mode

        if neuron_structure is None:
            self._neuron_structure = SingleCompartmentNeuron(1)
        else:
            if len(neuron_structure.compartments.get_compartments()) > 1:
                # Issue #4020 (we always record the first neuron circuit in
                # the first compartment)
                raise ValueError('Currently only neurons with a single '
                                 'compartment are supported.')
            self._neuron_structure = neuron_structure

    def register_hw_entity(self) -> None:
        """
        Infere neuron ids on hardware and register them.
        """
        self.unit_ids = np.arange(
            self.experiment.id_counter, self.experiment.id_counter + self.size)
        self.experiment.neuron_placement.register_id(
            self.unit_ids, self._neuron_structure.compartments,
            self._placement_constraint)
        self.experiment.id_counter += self.size
        self.experiment.register_population(self)

        # Handle offset
        if isinstance(self.offset, torch.Tensor):
            assert self.offset.shape[0] == self.size
        if isinstance(self.offset, dict):
            # Get populations HW neurons
            coords = self.experiment.neuron_placement.id2logicalneuron(
                self.unit_ids)
            offset = torch.zeros(self.size)
            for i, nrn in enumerate(coords):
                offset[i] = self.offset[nrn]
            self.offset = offset

        # Handle scale
        if isinstance(self.scale, torch.Tensor):
            assert self.scale.shape[0] == self.size
        if isinstance(self.scale, dict):
            # Get populations HW neurons
            coords = self.experiment.neuron_placement.id2logicalneuron(
                self.unit_ids)
            scale = torch.zeros(self.size)
            for i, nrn in enumerate(coords):
                scale[i] = self.scale[nrn]
            self.scale = scale

        if self._enable_madc_recording:
            if self.experiment.has_madc_recording:
                raise RuntimeError(
                    "Another HXModule already registered MADC recording. "
                    + "MADC recording is only enabled for a "
                    + "single neuron.")
            self.experiment.has_madc_recording = True
        log.TRACE(f"Registered hardware  entity '{self}'.")

    @property
    def mask(self) -> None:
        """
        Getter for spike mask.

        :returns: Returns the current spike mask.
        """
        return self._mask

    @mask.setter
    def mask(self, mask: torch.Tensor) -> None:
        """
        Setter for the spike mask.

        :param mask: Spike mask. Must be of shape `(self.size,)`.
        """
        # Mark dirty
        self._changed_since_last_run = True
        self._mask = mask

    @staticmethod
    def create_default_hw_entity() -> lola.AtomicNeuron:
        """
        At the moment, the default neuron is loaded from grenade's ChipConfig
        object, which holds the atomic neurons configured as a calibration is
        loaded in `hxtorch.hardware_init()`.

        TODO: - Needed?
              - Maybe this can return a default neuron, when pop-specific
                calibration is needed.
        """
        return lola.AtomicNeuron()

    def configure_hw_entity(self, neuron_id: int,
                            neuron_block: lola.NeuronBlock,
                            coord: halco.LogicalNeuronOnDLS) \
            -> lola.NeuronBlock:
        """
        Configures a neuron in the given layer with its specific properties.
        The neurons digital event outputs are enabled according to the given
        spiking mask.

        TODO: Additional parameterization should happen here, i.e. with
              population-specific parameters.

        :param neuron_id: In-population neuron index.
        :param neuron_block: The neuron block hardware entity.
        :param coord: Coordinate of neuron on hardware.
        :returns: Configured neuron block.
        """
        self._neuron_structure.implement_morphology(coord, neuron_block)
        self._neuron_structure.set_spike_recording(self.mask[neuron_id],
                                                   coord, neuron_block)
        if neuron_id == self._record_neuron_id:
            self._neuron_structure.enable_madc_recording(
                coord, neuron_block, self._madc_readout_source)
        return neuron_block

    def add_to_network_graph(self,
                             builder: grenade.NetworkBuilder) \
            -> grenade.PopulationDescriptor:
        """
        Add the layer's neurons to grenades network builder. If
        `enable_spike_recording` is enabled the neuron's spikes are recorded
        according to the layer's spiking mask. If no spiking mask is given all
        neuron spikes will be recorded. Note, the event output of the neurons
        are configured in `configure_hw_entity`.
        If `enable_cadc_recording` is enabled the populations neuron's are
        registered for CADC membrane recording.
        If `enable_madc_recording` is enabled the neuron with in-population
        index `record_neuron_id` will be recording via the MADC. Note, since
        the MADC can only record a single neuron on hardware, other Neuron
        layers registering also MADC recording might overwrite the setting
        here.

        :param builder: Grenade's network builder to add the layer's population
            to.
        :returns: Returns the builder with the population added.
        """
        # Create neuron mask if none is given (no dropout)
        if self._mask is None:
            self._mask = np.ones_like(self.unit_ids, dtype=bool)

        # Enable record spikes according to neuron mask
        if self._enable_spike_recording:
            enable_record_spikes = np.ones_like(self.unit_ids, dtype=bool)
        else:
            enable_record_spikes = np.zeros_like(self.unit_ids, dtype=bool)

        # get neuron coordinates
        coords: List[halco.LogicalNeuronOnDLS] = \
            self.experiment.neuron_placement.id2logicalneuron(self.unit_ids)

        # create receptors
        receptors = set([
            grenade.Receptor(
                grenade.Receptor.ID(),
                grenade.Receptor.Type.excitatory),
            grenade.Receptor(
                grenade.Receptor.ID(),
                grenade.Receptor.Type.inhibitory),
        ])

        neurons: List[grenade.Population.Neuron] = [
            grenade.Population.Neuron(
                logical_neuron,
                {halco.CompartmentOnLogicalNeuron():
                 grenade.Population.Neuron.Compartment(
                     grenade.Population
                     .Neuron.Compartment.SpikeMaster(
                         0, enable_record_spikes[i]), [receptors] * len(
                         logical_neuron.get_atomic_neurons()))})
            for i, logical_neuron in enumerate(coords)
        ]

        # create grenade population
        gpopulation = grenade.Population(neurons)

        # add to builder
        self.descriptor = builder.add(gpopulation)

        if self._enable_cadc_recording:
            for in_pop_id, unit_id in enumerate(self.unit_ids):
                neuron = grenade.CADCRecording.Neuron()
                neuron.population = self.descriptor
                neuron.neuron_on_population = in_pop_id
                neuron.compartment_on_neuron = 0
                neuron.atomic_neuron_on_compartment = 0
                self.experiment.cadc_recording[unit_id] = neuron

        # No recording registered -> return
        if not self._enable_madc_recording:
            return self.descriptor

        # add MADC recording
        # NOTE: If two populations register MADC reordings grenade should
        #       throw in the following
        madc_recording = grenade.MADCRecording()
        madc_recording.population = self.descriptor
        madc_recording.source = self._madc_readout_source
        madc_recording.neuron_on_population = int(self._record_neuron_id)
        madc_recording.compartment_on_neuron = \
            halco.CompartmentOnLogicalNeuron()
        madc_recording.atomic_neuron_on_compartment = 0
        builder.add(madc_recording)
        log.TRACE(f"Added population '{self}' to grenade graph.")

        return self.descriptor

    @staticmethod
    def add_to_input_generator(
            module: HXModule,
            builder: grenade.InputGenerator) -> None:
        """
        Add the input to an input module to grenades input generator.
        :param module: The module to add the input for.
        :param builder: Grenade's logical network builder.
        """

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
        spikes, cadc, madc = None, None, None

        # Get cadc samples
        if self._enable_cadc_recording:
            # Get dense representation
            cadc = hw_cadc.to_dense(
                self.experiment.dt, mode=self.interpolation_mode)

            # Shift CADC samples in time
            if self.cadc_time_shift != 0:
                cadc = torch.roll(cadc, shifts=-self.cadc_time_shift, dims=0)
            # If shift is to earlier times, we pad with last CADC value
            if self.cadc_time_shift > 0:
                cadc[-self.cadc_time_shift:] = \
                    cadc[-self.cadc_time_shift - 1].unsqueeze(0)
            # If shift is to later times, we pad with first CADC value
            if self.cadc_time_shift < 0:
                cadc[:-self.cadc_time_shift] = \
                    cadc[-self.cadc_time_shift].unsqueeze(0)

            # Offset CADC traces
            if self.shift_cadc_to_first:
                cadc = cadc - cadc[0].unsqueeze(0)
            else:
                cadc -= self.offset

            # Scale CADC traces
            cadc *= self.scale

        # Get spikes
        if self._enable_spike_recording:
            spikes = hw_spikes.to_dense(self.experiment.dt).float()

        # Get madc trace
        if self._enable_madc_recording:
            raise NotImplementedError(
                "MADCHandle to dense torch Tensor is not implemented yet.")

        return spikes, cadc, madc
