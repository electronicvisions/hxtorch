# pylint: disable=too-many-lines
"""
Implementing SNN modules
"""
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Dict,
    Tuple,
    Type,
    Optional,
    Union,
    List,
)
from warnings import warn
import pylogging as logger

import torch

from dlens_vx_v3 import halco, hal
import pygrenade_vx as grenade

import hxtorch.spiking.functional as F
from hxtorch.core.morphology import (
    Morphology,
    SingleCompartmentNeuron,
)
from hxtorch.core.parameter import (
    HXBaseParameter,
    HXTransformedModelParameter,
    MockParameter,
)
from hxtorch.core.utils.readout_source import ReadoutSource
from hxtorch.spiking.functional.mock import RandomNoise
from hxtorch.spiking.handle import (
    Handle,
    SynapseHandle,
    LIFObservables,
    LIObservables,
)
from hxtorch.spiking.observables import AnalogObservable
from hxtorch.spiking.modules.types.population import Population

if TYPE_CHECKING:
    from hxtorch.spiking.observables import HXTorchObservables
    from hxtorch.spiking.experiment import Experiment


class AELIF(Population):
    """
    Layer of neurons with configurable dynamics up to adaptive exponential
    leaky integrate-and-fire complexity.
    Neuron dynamics can be configured by enabling or disabling the firing
    behaviour and/or the leak, exponential, subthreshold- or spike-triggered
    adaptation term(s) from the differential equations of the adaptive
    exponential leaky integrate-and-fire model.
    If neither subthreshold adaptation nor spike-triggered adaptation are
    enabled, adaptation is not considered at all.

    Caveat:
    For execution on hardware, this module can only be used in conjunction with
    a preceding Synapse module.
    """

    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches,
    # pylint: disable=too-many-statements, invalid-name
    def __init__(
        self,
        size: int,
        experiment: Experiment,
        leak: HXBaseParameter = 80,
        reset: HXBaseParameter = 80,
        threshold: HXBaseParameter = 125,
        tau_mem: HXBaseParameter = 10e-6,
        tau_syn: HXBaseParameter = 10e-6,
        i_synin_gm: HXBaseParameter = 500,
        membrane_capacitance: HXBaseParameter = (
            HXTransformedModelParameter(
                10e-6, lambda model_value: int(model_value / 10e-6 * 63)
            )
        ),
        leak_conductance: Optional[HXBaseParameter] = None,
        refractory_time: HXBaseParameter = 1e-6,
        synapse_dac_bias: HXBaseParameter = 600,
        holdoff_time: HXBaseParameter = 0.,
        method: str = "superspike",
        alpha: float = 50.,
        exponential_slope: HXBaseParameter = 50e-3,
        exponential_threshold: HXBaseParameter = 110,
        subthreshold_adaptation_strength: HXBaseParameter = 1,
        spike_triggered_adaptation_increment: HXBaseParameter = 1,
        clock_scale_adaptation_pulse: Tuple[int] = (5, 5),
        tau_adap: HXBaseParameter = 100e-6,
        leak_adaptation: Optional[HXBaseParameter] = None,
        chip_coordinate: Optional[
            Tuple[grenade.common.ChipOnConnection,
                  grenade.common.ConnectionOnExecutor]] = None,
        enable_spike_recording: bool = True,
        enable_cadc_recording: bool = True,
        enable_cadc_recording_placement_in_dram: bool = False,
        cadc_readout_source: hal.NeuronConfig.ReadoutSource = (
            ReadoutSource.VOLTAGE),
        enable_madc_recording: bool = False,
        record_neuron_id: Optional[int] = None,
        madc_readout_source: hal.NeuronConfig.ReadoutSource = (
            ReadoutSource.VOLTAGE),
        placement_constraint: Optional[
            List[halco.LogicalNeuronOnDLS]] = None,
        trace_offset: Union[Dict[halco.LogicalNeuronOnDLS, float],
                            torch.Tensor, float] = 0.,
        trace_scale: Union[Dict[halco.LogicalNeuronOnDLS, float],
                           torch.Tensor, float] = 1.,
        cadc_time_shift: int = 0, shift_cadc_to_first: bool = False,
        interpolation_mode: str = "linear",
        neuron_structure: Optional[Morphology] = None,
        trace_noise_current: Optional[RandomNoise] = None,
        trace_noise_voltage: Optional[RandomNoise] = None,
        trace_noise_adaptation: Optional[RandomNoise] = None,
        cadc_readout_noise_current: Optional[RandomNoise] = None,
        cadc_readout_noise_voltage: Optional[RandomNoise] = None,
        cadc_readout_noise_adaptation: Optional[RandomNoise] = None,
        leaky: bool = True,
        fire: bool = True,
        exponential: bool = True,
        subthreshold_adaptation: bool = True,
        spike_triggered_adaptation: bool = True,
        **extra_params,
    ) -> None:
        """
        Initialize a neuron layer. This module creates a population of spiking
        neurons of size `size`. This module has an internal spiking mask, which
        allows to disable the event output and spike recordings of specific
        neurons within the layer. This is particularly useful for dropout.

        The neuron is parameterized by the `ModuleParameterType` parameters:
            leak, reset, threshold, tau_syn, tau_mem, i_synin_gm,
            membrane_capacitance, refractory_time, synapse_dac_bias,
            holdoff_time, exponential_slope, exponential_threshold, a, b,
            tau_adap.
        More infos to the respective parameters on BSS-2 can be found in
        `calix.spiking.neuron.NeuronCalibTarget`. If the parameters are not
        given as `ParameterType`, they are implicitly converted to
        `HXParameter` which provides the same value to the BSS-2 calibration
        (`param.hardware_value`) (and thus the hardware operation state) as to
        the numerical model (`param.model_value`) defined in `forward_func`.
        `MixedHXModelParameter` and `HXTransformedModelParameter` allow using
        different values on BSS-2 and in the numerics. This is useful if the
        dynamic range on hardware and in the numerical model differ. If so,
        the trace and weight scaling parameters need to be set accordingly in
        order to translate the weights to their corresponding hardware value
        and the hardware measurements into the dynamic range used in the
        numerics.

        :param size: Size of the population.
        :param experiment: Experiment to append layer to.
        :param leak: The leak potential. Defaults to HXParameter(80).
        :param reset: The reset potential. Defaults to HXParameter(80).
        :param threshold: The threshold potential. Defaults to
            HXParameter(125).
        :param tau_mem: The membrane time constant in s. Defaults to
            HXParameter(10e-6).
        :param tau_syn: The synaptic time constant in s. Defaults to
            HXParameter(10e-6).
        :param i_synin_gm: A hardware parameter adjusting the hardware neuron
            -specific synaptic efficacy. Defaults to HXParameter(500).
        :param membrane_capacitance: The capacitance of the membrane. The
            available range is 0 to approximately 2.2 pF, represented as 0 to
            63 LSB.
        :param leak_conductance: The leak conductance of the memebrane. When
            set to None, this value is set to membrane_capacitance / tau_mem.
            Defaults to None.
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
        :param exponential_slope: The exponential slope. Defaults to
            HxParameter(50e-3).
        :param exponential_threshold: The exponential threshold. Defaults to
            HxParameter(110).
        :param subthreshold_adaptation_strength: The subthreshold adaptation
            strength. Defaults to HxParameter(1).
        :param spike_triggered_adaptation_increment: The spike-triggered
            adaptation offset. Defaults to HxParameter(1).
        :param clock_scale_adaptation_pulse: This parameter controls the
            duration of the current pulse on hardware, that flows onto the
            capacitance of the adaptation. The set duration equals
            2^(`clock_scale_adaptation_pulse` + 1) / 250e6 seconds.
            The magnitude of the spike-triggered adaptation increment on
            hardware corresponds to the product of this duration and the
            hardware-value set for `spike_triggered_adaptation_increment`.
            This value can be set per hemisphere individually. Defaults to
            (5, 5).
        :param tau_adap: The adaptation time constant in s. Defaults to
            HxParameter(100e-6).
        :param leak_adaptation: A value for the leak potential of the membrane,
            which is taken into account by the subthreshold adaptation
            mechanism on hardware. It may distinguish from the setting of the
            actual leak potential. If value is `None`, the value of the actual
            leak potential is taken. Only applicable on hardware, not in
            simulation.
        :param execution_instance: Execution instance to place to.
        :param chip_coordinate: Chip coordinate this module is placed on.
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
            single neuron can be recorded. This membrane trace is sampled with
            a significantly higher resolution as with the CADC.
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
        :param trace_noise_current: `RandomNoise` object which generates
            random noise that is added onto the simulation result of the
            synaptic input current in each time step during simulation, in
            order to mock temporal noise. If set to `None`, no temporal noise
            will be applied to the current trace.
        :param trace_noise_voltage: `RandomNoise` object which generates
            random noise that is added onto the simulation result of the
            membrane voltage increment in each time step during simulation in
            order to mock temporal noise. If set to `None`, no temporal noise
            will be applied to the voltage trace.
        :param trace_noise_adaptation: `RandomNoise` object which generates
            random noise that is added onto the simulation result of the
            adaptation in each time step during simulation in order to mock
            temporal noise. If set to `None`, no temporal noise will be
            applied to the adaptation trace.
        :param cadc_readout_noise_current: `RandomNoise` object which generates
            random noise that is added onto the simulation result of the
            synaptic current once after the simulation in order to mock readout
            noise.  If set to `None`, no readout noise will be applied to the
            synaptic current.
        :param cadc_readout_noise_voltage: `RandomNoise` object which generates
            random noise that is added onto the simulation result of the
            membrane voltage once after the simulation in order to mock readout
            noise.  If set to `None`, no readout noise will be applied to the
            membrane voltage.
        :param cadc_readout_noise_adaptation: `RandomNoise` object which
            generates random noise that is added onto the simulation result of
            the adaptation once after the simulation in order to mock readout
            noise.  If set to `None`, no readout noise will be applied to the
            adaptation.
        :param leaky: Flag for enabling / disabling the leak term.
        :param fire: Flag for enabling / disabling the firing behaviour.
        :param exponential: Flag for enabling / disabling the exponential term.
        :param subthreshold_adaptation: Flag for enabling / disabling the
            subthreshold adaptation term.
        :param spike_triggered_adaptation: Flag for enabling / disabling the
            spike-triggered adaptation.
        """
        super().__init__(
            size,
            experiment=experiment,
            chip_coordinate=chip_coordinate,
            leak=leak,
            reset=reset,
            threshold=threshold,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            i_synin_gm=i_synin_gm,
            membrane_capacitance=membrane_capacitance,
            leak_conductance=leak_conductance,
            refractory_time=refractory_time,
            synapse_dac_bias=synapse_dac_bias,
            holdoff_time=holdoff_time,
            exponential_slope=exponential_slope,
            exponential_threshold=exponential_threshold,
            subthreshold_adaptation_strength=(
                subthreshold_adaptation_strength),
            spike_triggered_adaptation_increment=(
                spike_triggered_adaptation_increment
            ),
            clock_scale_adaptation_pulse=clock_scale_adaptation_pulse,
            tau_adap=tau_adap,
            leak_adaptation=leak_adaptation,
            enable_spike_recording=enable_spike_recording,
            enable_cadc_recording=enable_cadc_recording,
            enable_cadc_recording_placement_in_dram=(
                enable_cadc_recording_placement_in_dram
            ),
            cadc_readout_source=cadc_readout_source,
            enable_madc_recording=enable_madc_recording,
            record_neuron_id=record_neuron_id,
            madc_readout_source=madc_readout_source,
            placement_constraint=placement_constraint,
            neuron_structure=neuron_structure,
            leaky=leaky,
            fire=fire,
            exponential=exponential,
            subthreshold_adaptation=subthreshold_adaptation,
            spike_triggered_adaptation=spike_triggered_adaptation,
            **extra_params,
        )

        for param_name in self._parameters_defaults:
            param = getattr(self, self._param_name_mapping[param_name])
            if isinstance(param, MockParameter):
                param.mean = torch.as_tensor(
                    param.mean, dtype=torch.float32).expand(self.size)

        self.alpha = alpha
        self.method = method
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

        self.leaky = leaky
        self.fire = fire
        self.exponential = exponential
        self.subthreshold_adaptation = subthreshold_adaptation
        self.spike_triggered_adaptation = spike_triggered_adaptation
        self.adaptation = self.subthreshold_adaptation \
            or self.spike_triggered_adaptation

        self.logger = logger.get("hxtorch.spiking.modules.AELIF")

        # Specify output type
        if not self.fire and not self.adaptation:
            self.output_type = type(Handle(
                'membrane_cadc', 'membrane_madc', 'current'))
        elif not self.fire:
            self.output_type = type(Handle(
                'membrane_cadc', 'membrane_madc', 'current', 'adaptation_cadc',
                'adaptation_madc'))
        elif not self.adaptation:
            self.output_type = type(Handle(
                'membrane_cadc', 'membrane_madc', 'current', 'spikes'))
        else:
            self.output_type = type(Handle(
                'membrane_cadc', 'membrane_madc', 'current', 'adaptation_cadc',
                'adaptation_madc', 'spikes'))

        self.trace_noise_current = trace_noise_current
        self.trace_noise_voltage = trace_noise_voltage
        self.trace_noise_adaptation = trace_noise_adaptation
        self.noisy_traces = not (
            trace_noise_current is None and trace_noise_voltage is None
            and trace_noise_adaptation is None)
        self.cadc_readout_noise_current = cadc_readout_noise_current
        self.cadc_readout_noise_voltage = cadc_readout_noise_voltage
        self.cadc_readout_noise_adaptation = cadc_readout_noise_adaptation

    def extra_repr(self) -> str:
        """ Add additional information """
        reprs = ""
        if self.fire:
            reprs += f"alpha={self.alpha}, " \
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

    def add_to_input_data(
        self,
        experiment,
        snippet_begin_time,
        snippet_end_time,
    ):
        """
        Convert trace dict configs to tensors using the hardware mapping
        """
        def map_value(tensor, coord_map):
            inter_graph_hyper_edge_descriptors = experiment.mapped_topology\
                .get_reference().get_reference()\
                .inter_graph_hyper_edges_by_reference(self.descriptor)
            for inter_graph_hyper_edge_descriptor \
                    in inter_graph_hyper_edge_descriptors:
                compatible_vertex_descriptor = experiment.mapped_topology\
                    .get_reference().get_reference().links(
                        inter_graph_hyper_edge_descriptor)[0]
                compatible_inter_graph_hyper_edge_descriptors = experiment\
                    .mapped_topology.get_reference()\
                    .inter_graph_hyper_edges_by_reference(
                        compatible_vertex_descriptor)
                for compatible_inter_graph_hyper_edge_descriptor in \
                        compatible_inter_graph_hyper_edge_descriptors:
                    partitioned_vertex_descriptors = experiment\
                        .mapped_topology.get_reference().links(
                            compatible_inter_graph_hyper_edge_descriptor)
                    assert len(partitioned_vertex_descriptors) == 1
                    section = experiment.mapped_topology.get_reference().get(
                        partitioned_vertex_descriptors[0])
                    for section_mapping_descriptor in \
                            experiment.mapped_topology\
                            .inter_graph_hyper_edges_by_reference(
                                partitioned_vertex_descriptors[0]):
                        section_mapping = experiment.mapped_topology.get(
                            section_mapping_descriptor)
                        if isinstance(
                                section_mapping,
                                grenade.network.abstract.ChipMapping):
                            continue
                        for i, element in enumerate(
                                section.get_shape().get_elements()):
                            anchor = section_mapping.anchors[i]
                            coord = halco.LogicalNeuronOnDLS(
                                self._neuron_structure.compartments, anchor[1])
                            tensor[element.value[0]] = coord_map[coord]
            return tensor

        # Handle offset
        if isinstance(self.offset, torch.Tensor):
            assert self.offset.shape[0] == self.size
        if isinstance(self.offset, dict):
            self.offset = map_value(torch.zeros(self.size), self.offset)

        # Handle scale
        if isinstance(self.scale, torch.Tensor):
            assert self.scale.shape[0] == self.size
        if isinstance(self.scale, dict):
            self.scale = map_value(torch.zeros(self.size), self.scale)

        return super().add_to_input_data(
            experiment, snippet_begin_time, snippet_end_time)

    @property
    def cadc_readout_source(self) -> hal.NeuronConfig.ReadoutSource:
        return self._cadc_readout_source

    @cadc_readout_source.setter
    def cadc_readout_source(
        self,
        source: hal.NeuronConfig.ReadoutSource,
    ) -> None:
        self.chaned_input_data = True
        self._cadc_readout_source = source

    @property
    def madc_readout_source(self) -> hal.NeuronConfig.ReadoutSource:
        self.chaned_input_data = True
        return self._madc_readout_source

    @madc_readout_source.setter
    def madc_readout_source(
        self,
        source: hal.NeuronConfig.ReadoutSource,
    ) -> None:
        self._madc_readout_source = source

    @property
    def mask(self) -> Optional[torch.Tensor]:
        """
        Getter for spike mask.

        :returns: Returns the current spike mask.
        """
        return self._spike_mask

    @mask.setter
    def mask(self, mask: torch.Tensor) -> None:
        """
        Setter for the spike mask.

        :param mask: Spike mask. Must be of shape `(self.size,)`.
        """
        # Mark dirty
        self._changed_since_last_run = True
        if self._enable_spike_recording:
            self._spike_mask = mask

    def post_process(self, hw_data: HXTorchObservables, runtime: float) \
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

        :return: Returns a handle containing the post processed data.
        """
        if not self.fire:
            assert not self._enable_spike_recording

        spikes, cadc, madc = None, None, None

        # TODO: unit of runtime

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

        voltage = AnalogObservable()
        adaptation = AnalogObservable()
        if self._cadc_readout_source == ReadoutSource.VOLTAGE:
            voltage.cadc = cadc
        elif self._cadc_readout_source == ReadoutSource.ADAPTATION:
            adaptation.cadc = cadc
        else:
            self.logger.ERROR(
                "Post processing for CADC readout source "
                + f"{self._cadc_readout_source} is not implemented yet."
            )
        if self._madc_readout_source == ReadoutSource.VOLTAGE:
            voltage.madc = madc
        elif self._madc_readout_source == ReadoutSource.ADAPTATION:
            adaptation.madc = madc
        else:
            self.logger.ERROR(
                "Post processing for MADC readout source "
                + f"{self._madc_readout_source} is not implemented yet.",
            )

        return Handle(voltage=voltage, adaptation=adaptation, spikes=spikes)

    # pylint: disable=redefined-builtin, invalid-name
    def forward_func(self, *input: SynapseHandle,
                     hw_data: Optional[type(Handle(
                         'voltage', 'adaptation', 'spikes'))] = None):
        """
        Execute forward function of the neuron layer according to the dynamics
        specified upon construction.

        :param input: SynapseHandle from preceding Synapse Module containing
            graded spikes which are used as input for the neuron layer.
        :param hw_data: Can contain result data from hardware run which is
            injected into the calculations.
        :returns: Returns dynamically constructed handle which contains the
            observable data resulting from the calculations.
            Depending on the neuron dynamics specified upon construction,
            it contains voltage, voltage_madc, current, adaptation (if
            adaptation is enabled) and spikes (if firing behaviour is
            enabled).
            All of these hardware observables are of type
            Optional[torch.Tensor] and have shape
            [time steps, batch size, population size]
        """
        refractory = True
        if (isinstance(self.refractory_time.model_value, torch.Tensor)
                and torch.equal(
                    self.refractory_time.model_value,
                    torch.zeros_like(self.refractory_time.model_value))
                or not isinstance(
                    self.refractory_time.model_value, torch.Tensor)
                and float(self.refractory_time.model_value) == 0.):
            refractory = False
        if refractory and not self.fire:
            warn("Refractory period is not applicable due to "
                 "specified non-spiking behaviour.",
                 category=UserWarning)
            refractory = False
        hw_voltage_cadc_trace_available = False
        hw_adaptation_cadc_trace_available = False
        hw_spikes_available = False
        if hw_data:
            hw_voltage_cadc_trace_available = hw_data.voltage.cadc is not None
            hw_adaptation_cadc_trace_available = \
                hw_data.adaptation.cadc is not None
            hw_spikes_available = hw_data.spikes is not None
        integration_step_code = F.CuBaStepCode(
            leaky=self.leaky,
            fire=self.fire,
            refractory=refractory,
            exponential=self.exponential,
            subthreshold_adaptation=self.subthreshold_adaptation,
            spike_triggered_adaptation=self.spike_triggered_adaptation,
            hw_voltage_trace_available=hw_voltage_cadc_trace_available,
            hw_adaptation_trace_available=hw_adaptation_cadc_trace_available,
            hw_spikes_available=hw_spikes_available,
            noisy_traces=self.noisy_traces).generate()
        assert all(synapse_handle.graded_spikes is not None for
                   synapse_handle in input)
        (membrane_cadc, membrane_madc, current, adaptation_cadc,
            adaptation_madc, spikes) = F.cuba_aelif_integration(
                tuple(
                    synapse_handle.graded_spikes for synapse_handle in input),
                leak=self.leak.model_value,
                reset=self.reset.model_value,
                threshold=self.threshold.model_value,
                tau_syn=self.tau_syn.model_value,
                c_mem=self.membrane_capacitance.model_value,
                g_l=(self.leak_conductance.model_value
                     if self.leak_conductance.model_value is not None
                     else self.membrane_capacitance.model_value_detach()
                     / self.tau_mem.model_value_detach()),
                refractory_time=self.refractory_time.model_value,
                method=self.method,
                alpha=self.alpha,
                exp_slope=self.exponential_slope.model_value,
                exp_threshold=self.exponential_threshold.model_value,
                subthreshold_adaptation_strength=(
                    self.subthreshold_adaptation_strength.model_value),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment.model_value),
                tau_adap=self.tau_adap.model_value,
                hw_data=hw_data,
                dt=self.experiment.dt,
                trace_noise_current=self.trace_noise_current,
                trace_noise_voltage=self.trace_noise_voltage,
                trace_noise_adaptation=self.trace_noise_adaptation,
                cadc_readout_noise_current=self.cadc_readout_noise_current,
                cadc_readout_noise_voltage=self.cadc_readout_noise_voltage,
                cadc_readout_noise_adaptation=(
                    self.cadc_readout_noise_adaptation),
                leaky=self.leaky,
                fire=self.fire,
                refractory=refractory,
                exponential=self.exponential,
                subthreshold_adaptation=self.subthreshold_adaptation,
                spike_triggered_adaptation=self.spike_triggered_adaptation,
                integration_step_code=integration_step_code)
        if not self.fire and not self.adaptation:
            return Handle(
                membrane_cadc=membrane_cadc, membrane_madc=membrane_madc,
                current=current)
        if not self.fire:
            return Handle(
                membrane_cadc=membrane_cadc, membrane_madc=membrane_madc,
                current=current, adaptation_cadc=adaptation_cadc,
                adaptation_madc=adaptation_madc)
        if not self.adaptation:
            return Handle(
                membrane_cadc=membrane_cadc, membrane_madc=membrane_madc,
                current=current, spikes=spikes)
        return Handle(
            membrane_cadc=membrane_cadc, membrane_madc=membrane_madc,
            current=current, adaptation_cadc=adaptation_cadc,
            adaptation_madc=adaptation_madc, spikes=spikes)


class LIF(AELIF):
    """
    Layer of leaky integrate-and-fire neurons.

    Caveat:
    For execution on hardware, this module can only be used in conjunction with
    a preceding Synapse module.
    """

    output_type: Type = LIFObservables

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(
        self,
        size: int,
        experiment: Experiment,
        leak: HXBaseParameter = 80,
        reset: HXBaseParameter = 80,
        threshold: HXBaseParameter = 125,
        tau_mem: HXBaseParameter = 10e-6,
        tau_syn: HXBaseParameter = 10e-6,
        i_synin_gm: HXBaseParameter = 500,
        membrane_capacitance: HXBaseParameter = (
            HXTransformedModelParameter(
                10e-6,
                lambda model_value: int(model_value / 10e-6 * 63),
            )
        ),
        leak_conductance: Optional[HXBaseParameter] = None,
        refractory_time: HXBaseParameter = 1e-6,
        synapse_dac_bias: HXBaseParameter = 600,
        holdoff_time: HXBaseParameter = 0e-6,
        method: str = "superspike",
        alpha: float = 50.,
        enable_spike_recording: bool = True,
        enable_cadc_recording: bool = True,
        enable_cadc_recording_placement_in_dram: bool = False,
        enable_madc_recording: bool = False,
        record_neuron_id: Optional[int] = None,
        placement_constraint: Optional[List[halco.LogicalNeuronOnDLS]] = None,
        trace_offset: Union[Dict[halco.AtomicNeuronOnDLS, float],
                            torch.Tensor, float] = 0.,
        trace_scale: Union[Dict[halco.AtomicNeuronOnDLS, float],
                           torch.Tensor, float] = 1.,
        cadc_time_shift: int = 0, shift_cadc_to_first: bool = False,
        interpolation_mode: str = "linear",
        neuron_structure: Optional[Morphology] = None,
        **extra_params,
    ) -> None:
        """
        Initialize a layer of leaky integrate-and-fire neurons.
        This module creates a population of spiking neurons of size `size`
        and has an internal spiking mask, which allows to disable the
        event output and spike recordings of specific neurons within the
        layer. This is particularly useful for dropout.


        The leaky integrate-and-fire neuron is parameterized by the
        `ModuleParameterType` parameters:
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
        dynamic range on hardware and in the numerical model differ. If so,
        the trace and weight scaling parameters need to be set accordingly in
        order to translate the weights to their corresponding hardware value
        and the hardware measurements into the dynamic range used in the
        numerics.

        :param size: Size of the population.
        :param experiment: Experiment to append layer to.
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
        :param leak_conductance: The leak conductance of the memebrane. When
            set to None, this value is set to membrane_capacitance / tau_mem.
            Defaults to None.
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
            single neuron can be recorded. This membrane trace is sampled with
            a significantly higher resolution as with the CADC.
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
        super().__init__(
            size,
            experiment,
            leak=leak,
            reset=reset,
            threshold=threshold,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            i_synin_gm=i_synin_gm,
            membrane_capacitance=membrane_capacitance,
            leak_conductance=leak_conductance,
            refractory_time=refractory_time,
            synapse_dac_bias=synapse_dac_bias,
            holdoff_time=holdoff_time,
            method=method,
            alpha=alpha,
            enable_spike_recording=enable_spike_recording,
            enable_cadc_recording=enable_cadc_recording,
            enable_cadc_recording_placement_in_dram=(
                enable_cadc_recording_placement_in_dram
            ),
            enable_madc_recording=enable_madc_recording,
            record_neuron_id=record_neuron_id,
            placement_constraint=placement_constraint,
            trace_offset=trace_offset, trace_scale=trace_scale,
            cadc_time_shift=cadc_time_shift,
            shift_cadc_to_first=shift_cadc_to_first,
            interpolation_mode=interpolation_mode,
            neuron_structure=neuron_structure,
            leaky=True,
            fire=True,
            exponential=False,
            subthreshold_adaptation=False,
            spike_triggered_adaptation=False,
            **extra_params)


class EventPropLIF(LIF):

    output_type: Type = LIFObservables

    # pylint: disable=redefined-builtin
    def forward_func(self, *input: SynapseHandle,
                     hw_data: Optional[Tuple[Optional[torch.Tensor]]] = None) \
            -> output_type:
        # EA 2025-01-30: No keywords allowed in apply
        # TODO: Enable multiple inputs
        assert len(input) == 1
        results = F.EventPropLIFFunction.apply(
            input[0].graded_spikes,
            self.leak.model_value,
            self.reset.model_value,
            self.threshold.model_value,
            self.tau_syn.model_value,
            self.tau_mem.model_value,
            hw_data,
            self.experiment.dt)
        return Handle(membrane_cadc=results[1], membrane_madc=None,
                      current=results[2], spikes=results[0])


# WARNING: THIS CLASS IS DEPRECATED
class NeuronExp(LIF):
    """
    Neuron layer with exponential Euler integration scheme.
    Synaptic and membrane time constant are required to be provided
    as HXTransformedModelParameter(exp(-dt/tau), -dt/ln(tau)).
    This ensures that the correct model and hardware values are being used.
    """

    # pylint: disable=redefined-builtin, arguments-differ
    def forward_func(self, input: SynapseHandle,
                     hw_data: Optional[Tuple[torch.Tensor]] = None) \
            -> LIFObservables:
        warn("The NeuronExp Module is deprecated!",
             category=DeprecationWarning)
        return LIFObservables(*F.exp_cuba_lif_integration(
            input.graded_spikes,
            leak=self.leak.model_value,
            reset=self.reset.model_value,
            threshold=self.threshold.model_value,
            tau_syn_exp=self.tau_syn.model_value,
            tau_mem_exp=self.tau_mem.model_value,
            method=self.method,
            alpha=self.alpha,
            hw_data=hw_data))


class LI(AELIF):
    """
    Layer of leaky integrator neurons

    Caveat:
    For execution on hardware, this module can only be used in conjunction with
    a preceding Synapse module.
    """

    output_type: Type = LIObservables

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(
        self,
        size: int,
        experiment: Experiment,
        leak: HXBaseParameter = 80,
        tau_mem: HXBaseParameter = 10e-6,
        tau_syn: HXBaseParameter = 10e-6,
        i_synin_gm: HXBaseParameter = 500,
        membrane_capacitance: HXBaseParameter = (
            HXTransformedModelParameter(
                10e-6,
                lambda model_value: int(model_value / 10e-6 * 63),
            )),
        leak_conductance: Optional[HXBaseParameter] = None,
        synapse_dac_bias: HXBaseParameter = 600,
        enable_cadc_recording: bool = True,
        enable_cadc_recording_placement_in_dram: bool = False,
        enable_madc_recording: bool = False,
        record_neuron_id: Optional[int] = None,
        placement_constraint: Optional[List[halco.LogicalNeuronOnDLS]] = None,
        trace_offset: Union[Dict[halco.AtomicNeuronOnDLS, float],
                            torch.Tensor, float] = 0.,
        trace_scale: Union[Dict[halco.AtomicNeuronOnDLS, float],
                           torch.Tensor, float] = 1.,
        cadc_time_shift: int = 0, shift_cadc_to_first: bool = False,
        interpolation_mode: str = "linear",
        neuron_structure: Optional[Morphology] = None,
        **extra_params,
    ) -> None:
        """
        Initialize a layer of leaky integrator neurons. This module creates a
        population of non-spiking neurons of size `size` and is equivalent to
        LIF when its spiking mask is disabled for all neurons.

        The leaky integrator neuron is parameterized by the
        `ModuleParameterType` parameters:
            leak, tau_mem, tau_syn, i_synin_gm, membrane_capacitance,
            synapse_dac_bias
        More infos to the respective parameters on BSS-2 can be found in
        `calix.spiking.neuron.NeuronCalibTarget`. If the parameters are not
        given as `ParameterType`, they are implicitly converted to
        `HXParameter` which provides the same value to the BSS-2 calibration
        (`param.hardware_value`) (and thus the hardware operation state) as to
        the numerical model (`param.model_value`) defined in `forward_func`.
        `MixedHXModelParameter` and `HXTransformedModelParameter` allow using
        different values on BSS-2 and in the numerics. This is useful if the
        dynamic range on hardware and in the numerical model differ. If so,
        the trace and weight scaling parameters need to be set accordingly in
        order to translate the weights to their corresponding hardware value
        and the hardware measurements into the dynamic range used in the
        numerics.

        :param size: Size of the population.
        :param experiment: Experiment to register the module in.
        :param leak: The leak potential. Defaults to HXParameter(80).
        :param tau_syn: The synaptic time constant in s. Defaults to
            HXParameter(10e-6).
        :param tau_mem: The membrane time constant in s. Defaults to
            HXParameter(10e-6).
        :param i_synin_gm: A hardware parameter adjusting the hardware neuron
            -specific synaptic efficacy. Defaults to HXParameter(500).
        :param membrane_capacitance: The capacitance of the membrane. The
            available range is 0 to approximately 2.2 pF, represented as 0 to
            63 LSB.
        :param leak_conductance: The leak conductance of the memebrane. When
            set to None, this value is set to membrane_capacitance / tau_mem.
            Defaults to None.
        :param synapse_dac_bias: Synapse DAC bias current that is desired. Can
            be lowered in order to reduce the amplitude of a spike at the input
            of the synaptic input OTA. This can be useful to avoid saturation
            when using larger synaptic time constants. Defaults to
            HXParameter(600).
        :param enable_cadc_recording: Enables or disables parallel sampling of
            the populations membrane trace via the CADC. A maximum sample rate
            of 1.7us is possible.
        :param enable_cadc_recording_placement_in_dram: Whether to place CADC
            recording data into DRAM (period ~6us) or SRAM (period ~2us).
        :param enable_madc_recording: Enables or disables the recording of the
            neurons `record_neuron_id` membrane trace via the MADC. Only a
            single neuron can be recorded. This membrane trace is sampled with
            a significantly higher resolution as with the CADC.
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
            size,
            experiment,
            leak=leak,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            i_synin_gm=i_synin_gm,
            membrane_capacitance=membrane_capacitance,
            leak_conductance=leak_conductance,
            refractory_time=0.,
            synapse_dac_bias=synapse_dac_bias,
            enable_spike_recording=False,
            enable_cadc_recording=enable_cadc_recording,
            enable_cadc_recording_placement_in_dram=(
                enable_cadc_recording_placement_in_dram
            ),
            enable_madc_recording=enable_madc_recording,
            record_neuron_id=record_neuron_id,
            placement_constraint=placement_constraint,
            trace_offset=trace_offset, trace_scale=trace_scale,
            cadc_time_shift=cadc_time_shift,
            shift_cadc_to_first=shift_cadc_to_first,
            interpolation_mode=interpolation_mode,
            neuron_structure=neuron_structure,
            leaky=True,
            fire=False,
            exponential=False,
            subthreshold_adaptation=False,
            spike_triggered_adaptation=False,
            **extra_params)


# WARNING: THIS CLASS IS DEPRECATED
class ReadoutNeuronExp(LI):
    """
    Neuron layer with exponential Euler integration scheme.
    Synaptic and membrane time constant are required to be provided
    as HXTransformedModelParameter(exp(-dt/tau), -dt/ln(tau)).
    This ensures that the correct model and hardware values are being used.
    """

    # pylint: disable=redefined-builtin
    def forward_func(self, *input: SynapseHandle,
                     hw_data: Optional[Tuple[torch.Tensor]] = None) \
            -> LIObservables:
        warn("The ReadoutNeuronExp Module is deprecated!",
             category=DeprecationWarning)
        return LIObservables(*F.exp_cuba_li_integration(
            tuple(handle.graded_spikes for handle in input),
            leak=self.leak.model_value,
            tau_syn_exp=self.tau_syn.model_value,
            tau_mem_exp=self.tau_mem.model_value,
            hw_data=hw_data))
