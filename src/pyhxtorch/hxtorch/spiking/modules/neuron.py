"""
Implementing SNN modules
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple, Type, Optional, Union, List
import pylogging as logger
import numpy as np

import torch

from dlens_vx_v3 import lola, hal, halco
import pygrenade_vx as grenade

import hxtorch.spiking.functional as F
from hxtorch.spiking.morphology import Morphology, SingleCompartmentNeuron
from hxtorch.spiking.handle import NeuronHandle, SynapseHandle
from hxtorch.spiking.modules.types import Population, ModuleParameterType
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment
    from hxtorch.spiking.observables import HardwareObservables
    from hxtorch.spiking.execution_instance import ExecutionInstance
    from pyhalco_hicann_dls_vx_v3 import DLSGlobal

log = logger.get("hxtorch.spiking.modules")


class Neuron(Population):
    """
    Neuron layer

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
    def __init__(self, size: int,
                 experiment: Experiment,
                 leak: ModuleParameterType = 80,
                 reset: ModuleParameterType = 80,
                 threshold: ModuleParameterType = 125,
                 tau_mem: ModuleParameterType = 10e-6,
                 tau_syn: ModuleParameterType = 10e-6,
                 i_synin_gm: ModuleParameterType = 500,
                 membrane_capacitance: ModuleParameterType = 63,
                 refractory_time: ModuleParameterType = 1e-6,
                 synapse_dac_bias: ModuleParameterType = 600,
                 holdoff_time: ModuleParameterType = 0e-6,
                 method: str = "superspike",
                 alpha: float = 50.,
                 execution_instance: Optional[ExecutionInstance] = None,
                 chip_coordinate: Optional[DLSGlobal] = None,
                 enable_spike_recording: bool = True,
                 enable_cadc_recording: bool = True,
                 enable_cadc_recording_placement_in_dram: bool = False,
                 enable_madc_recording: bool = False,
                 record_neuron_id: Optional[int] = None,
                 placement_constraint: Optional[
                     List[halco.LogicalNeuronOnDLS]] = None,
                 trace_offset: Union[Dict[halco.LogicalNeuronOnDLS, float],
                                     torch.Tensor, float] = 0.,
                 trace_scale: Union[Dict[halco.LogicalNeuronOnDLS, float],
                                    torch.Tensor, float] = 1.,
                 cadc_time_shift: int = 0, shift_cadc_to_first: bool = False,
                 interpolation_mode: str = "linear",
                 neuron_structure: Optional[Morphology] = None,
                 **extra_params) -> None:
        """
        Initialize a Neuron. This module creates a population of spiking
        neurons of size `size`. This module has a internal spiking mask, which
        allows to disable the event output and spike recordings of specific
        neurons within the layer. This is particularly useful for dropout.

        The neuron is parameterized by the `ModuleParameterType`d parameters:
            leak, reset, threshold, tau_mem, tau_syn, i_synin_gm,
            membrane_capacitance, refractory_time, synapse_dac_bias,
            holdoff_time.
        More infos to the respective parameters on BSS-2 can be found in
        `calix.spiking.neuron.NeuronCalibTarget`. If the parameters are not
        given as `ParameterType`, they are implicitly converted to
        `HXParameter` which provides the same value to the BSS-2 calibration
        (`param.hardware_value`) (and thus the hardware operation state) as to
        the numerical model (`param.model_value`) defined in `forward_func`.
        `MixedHXModelParameter` and `HXTransformedModelParameter` allow using
        different values on BSS-2 and in the numerics. This is useful if the
        dynamic range on hardware and in the numerical model  differ. If so,
        the trace and weight scaling parameters need to be set accordingly in
        order to translate the weights to their corresponding hardware value
        and the hardware measurements into the dynamic range used in the
        numerics.

        :param size: Size of the population.
        :param experiment: Experiment to append layer to.
        :param execution_instance: Execution instance to place to.
        :param chip_coordinate: Chip coordinate this module is placed on.
        :param leak: The leak potential. Defaults to HXParameter(80).
        :param reset: The reset potential. Defaults to HXParameter(80).
        :param threshold: The threshold potential. Defaults to
            HXParameter(125).
        :param tau_syn: The synaptic time constant in s. Defaults to
            HXParameter(10e-6).
        :param tau_mem: The membrane time constant in s. Defaults to
            HXParameter(10e-6).
        :param i_synin_gm: A hardware parameter adjusting the hardware neuron
            -specific synaptic efficacy. Defaults to HXParameter(500).
        :param membrane_capacitance: The capacitance of the membrane. The
            available range is 0 to approximately 2.2 pF, represented as 0 to
            63 LSB.
        :param refractory_time: The refractory time constant in s. Defaults to
            HXParameter(1e-6).
        :param synapse_dac_bias: Synapse DAC bias current that is desired. Can
            be lowered in order to reduce the amplitude of a spike at the input
            of the synaptic input OTA. This can be useful to avoid saturation
            when using larger synaptic time constants. Defaults to
            HXParameter(600).
        :param holdoff_time: Target length of the holdoff period in s. The
            holdoff period is the time at the end of the refractory period in
            which the clamping to the reset voltage is already released but new
            spikes can still not be generated. Defaults to HXParameter(0e-6).
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
        super().__init__(size,
                         experiment=experiment,
                         execution_instance=execution_instance,
                         chip_coordinate=chip_coordinate,
                         leak=leak,
                         reset=reset,
                         threshold=threshold,
                         tau_mem=tau_mem,
                         tau_syn=tau_syn,
                         i_synin_gm=i_synin_gm,
                         membrane_capacitance=membrane_capacitance,
                         refractory_time=refractory_time,
                         synapse_dac_bias=synapse_dac_bias,
                         holdoff_time=holdoff_time,
                         **extra_params)

        if placement_constraint is not None \
                and len(placement_constraint) != size:
            raise ValueError(
                "The number of neurons in logical neurons in "
                + "`hardware_constraints` does not equal the `size` of the "
                + "module.")

        self.alpha = alpha
        self.method = method

        self._enable_spike_recording = enable_spike_recording
        self._enable_cadc_recording = enable_cadc_recording
        self._enable_cadc_rec_in_dram = \
            enable_cadc_recording_placement_in_dram
        self._enable_madc_recording = enable_madc_recording
        self._record_neuron_id = record_neuron_id
        self._placement_constraint = placement_constraint
        self._mask: Optional[torch.Tensor] = None

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

    def extra_repr(self) -> str:
        """ Add additional information """
        reprs = f"alpha={self.alpha}, " \
            + f"method={self.method}, "
        if not self.experiment.mock:
            reprs += f"spike_recording={self._enable_spike_recording}, " \
                + f"cadc_recording={self._enable_cadc_recording}, " \
                + f"madc_recording={self._enable_madc_recording}, " \
                + f"record_neuron_id={self._record_neuron_id}, " \
                + f"trace_scale={self.scale}, " \
                + f"trace_offset={self.offset}, " \
                + f"cadc_time_shift={self.cadc_time_shift}, " \
                + f"shift_cadc_to_first={self.shift_cadc_to_first}, " \
                + f"interpolation_mode={self.interpolation_mode}, " \
                + f"neuron_structure={self._neuron_structure}, "
        reprs += f"{super().extra_repr()}"
        return reprs

    def register_hw_entity(self) -> None:
        """
        Infer neuron IDs on hardware and register them.
        """
        self.unit_ids = np.arange(
            self.execution_instance.id_counter,
            self.execution_instance.id_counter + self.size)
        self.execution_instance.neuron_placement.register_id(
            self.unit_ids, self._neuron_structure.compartments,
            self._placement_constraint)
        self.execution_instance.id_counter += self.size
        self.experiment.register_population(self)

        # Handle offset
        if isinstance(self.offset, torch.Tensor):
            assert self.offset.shape[0] == self.size
        if isinstance(self.offset, dict):
            # Get populations HW neurons
            coords = self.execution_instance.neuron_placement \
                .id2logicalneuron(self.unit_ids)
            offset = torch.zeros(self.size)
            for i, nrn in enumerate(coords):
                offset[i] = self.offset[nrn]
            self.offset = offset

        # Handle scale
        if isinstance(self.scale, torch.Tensor):
            assert self.scale.shape[0] == self.size
        if isinstance(self.scale, dict):
            # Get populations HW neurons
            coords = self.execution_instance.neuron_placement \
                .id2logicalneuron(self.unit_ids)
            scale = torch.zeros(self.size)
            for i, nrn in enumerate(coords):
                scale[i] = self.scale[nrn]
            self.scale = scale

        if self._enable_madc_recording:
            if self.execution_instance.has_madc_recording:
                raise RuntimeError(
                    "Another HXModule already registered MADC recording. "
                    + "MADC recording is only enabled for a "
                    + "single neuron on one execution instance.")
            self.execution_instance.has_madc_recording = True

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
                             builder: grenade.network.NetworkBuilder) \
            -> grenade.network.PopulationOnNetwork:
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
        coords: List[halco.LogicalNeuronOnDLS] = self.execution_instance \
            .neuron_placement.id2logicalneuron(self.unit_ids)

        # create receptors
        receptors = set([
            grenade.network.Receptor(
                grenade.network.Receptor.ID(),
                grenade.network.Receptor.Type.excitatory),
            grenade.network.Receptor(
                grenade.network.Receptor.ID(),
                grenade.network.Receptor.Type.inhibitory),
        ])

        neurons: List[grenade.network.Population.Neuron] = [
            grenade.network.Population.Neuron(
                logical_neuron,
                {halco.CompartmentOnLogicalNeuron():
                 grenade.network.Population.Neuron.Compartment(
                     grenade.network.Population
                     .Neuron.Compartment.SpikeMaster(
                         0, enable_record_spikes[i]), [receptors] * len(
                         logical_neuron.get_atomic_neurons()))})
            for i, logical_neuron in enumerate(coords)
        ]

        # create grenade population
        gpopulation = grenade.network.Population(neurons,
                                                 self.chip_coordinate)

        # add to builder
        self.descriptor = builder.add(
            gpopulation, self.execution_instance.ID)

        if self._enable_cadc_recording:
            for in_pop_id, unit_id in enumerate(self.unit_ids):
                neuron = grenade.network.CADCRecording.Neuron()
                neuron.coordinate.population = self.descriptor
                neuron.source = self._cadc_readout_source
                neuron.coordinate.neuron_on_population = in_pop_id
                neuron.coordinate.compartment_on_neuron = 0
                neuron.coordinate.atomic_neuron_on_compartment = 0
                if unit_id not in self.execution_instance.cadc_neurons:
                    self.execution_instance.cadc_neurons.update({unit_id: []})
                self.execution_instance.cadc_neurons.update({
                    unit_id:
                    self.execution_instance.cadc_neurons[unit_id] + [neuron]})
            if self.execution_instance.record_cadc_into_dram \
                is not None and \
                self.execution_instance.record_cadc_into_dram \
                    != self._enable_cadc_rec_in_dram:
                raise RuntimeError(
                    "Requesting CADC DRAM and SRAM recording simultaneously.")
            self.execution_instance.record_cadc_into_dram = \
                self._enable_cadc_rec_in_dram

        # No recording registered -> return
        if not self._enable_madc_recording:
            return self.descriptor

        # add MADC recording
        # NOTE: If two populations register MADC recordings grenade should
        #       throw in the following
        madc_recording_neuron = grenade.network.MADCRecording.Neuron()
        madc_recording_neuron.coordinate.population = self.descriptor
        madc_recording_neuron.source = self._madc_readout_source
        madc_recording_neuron.coordinate.neuron_on_population = int(
            self._record_neuron_id)
        madc_recording_neuron.coordinate.compartment_on_neuron = \
            halco.CompartmentOnLogicalNeuron()
        madc_recording_neuron.coordinate.atomic_neuron_on_compartment = 0
        madc_recording = grenade.network.MADCRecording([madc_recording_neuron],
                                                       self.chip_coordinate)
        builder.add(madc_recording, self.execution_instance.ID)
        log.TRACE(f"Added population '{self}' to grenade graph.")

        return self.descriptor

    def post_process(self, hw_data: HardwareObservables, runtime: float) \
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

        :param hw_data: A ``HardwareObservables`` instance holding the
            population's recorded hardware observables.
        :param runtime: The requested runtime of the experiment on hardware in
            s.

        :return: Returns a tuple of optional torch.Tensors holding the hardware
            data (madc or cadc,)
        """
        spikes, cadc, madc = None, None, None

        # Get cadc samples
        if self._enable_cadc_recording:
            # Get dense representation
            cadc = hw_data.cadc.to_dense(
                runtime, self.experiment.dt, mode=self.interpolation_mode)

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
            spikes = hw_data.spikes.to_dense(
                runtime, self.experiment.dt).float()

        # Get madc trace
        if self._enable_madc_recording:
            madc = hw_data.madc.to_raw()

        return spikes, cadc, madc

    # pylint: disable=redefined-builtin
    def forward_func(self, input: SynapseHandle,
                     hw_data: Optional[Tuple[torch.Tensor]] = None) \
            -> NeuronHandle:
        return NeuronHandle(*F.cuba_lif_integration(
            input.graded_spikes,
            leak=self.leak.model_value,
            reset=self.reset.model_value,
            threshold=self.threshold.model_value,
            tau_syn=self.tau_syn.model_value,
            tau_mem=self.tau_mem.model_value,
            method=self.method,
            alpha=self.alpha,
            hw_data=hw_data,
            dt=self.experiment.dt))


class EventPropNeuron(Neuron):
    # pylint: disable=redefined-builtin
    def forward_func(self, input: SynapseHandle,
                     hw_data: Optional[Tuple[torch.Tensor]] = None) \
            -> NeuronHandle:
        # EA 2025-01-30: No keywords allowed in apply
        return NeuronHandle(*F.EventPropNeuronFunction.apply(
            input.graded_spikes,
            self.leak.model_value,
            self.reset.model_value,
            self.threshold.model_value,
            self.tau_syn.model_value,
            self.tau_mem.model_value,
            hw_data,
            self.experiment.dt))
