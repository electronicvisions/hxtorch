""" Define base population and projection classes for BSS-2 """
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Union, List
from functools import partial

import numpy as np

import pyccalix  # noqa: F401 # needed to register ccalix::TimeInS type
from pyccalix import TimeInS, CapacitanceInFarad

from dlens_vx_v3 import hal, lola, halco
import pygrenade_vx as grenade
from pygrenade_vx.network.abstract.frontend import (
    ExperimentElement,
    ExperimentSnippet,
)
import pygrenade_vx.network.abstract as gabstract
import pygrenade_vx as grenade_vx


from hxtorch import logger
from pyhalco_hicann_dls_vx_v3 import CompartmentOnLogicalNeuron

from hxtorch.core.modules.hx_module import HXBaseModule
from hxtorch.core.utils.readout_source import ReadoutSource
from hxtorch.core.parameter import (
    HXBaseParameter,
    HXParameter,
    ParameterType,
)
from hxtorch.core.morphology import (
    Morphology,
    SingleCompartmentNeuron,
)
from hxtorch.core.observables import HXObservables
from hxtorch.core.experiment import BaseExperiment

if TYPE_CHECKING:
    from pyhalco_hicann_dls_vx_v3 import DLSGlobal

ModuleParameterType = Union[ParameterType, float, int]


class BasePopulation(
    HXBaseModule,
    ExperimentElement,
):
    """ Base class for populations """
    __constants__ = ['size']
    size: int
    _observables_factory = HXObservables

    def __init__(
        self, size: int,
        experiment: BaseExperiment,
        chip_coordinate: Dict[
            grenade.common.ExecutionInstanceID,
            DLSGlobal
        ] | None = None,
    ) -> None:
        """
        :param size: Number of input neurons.
        :param experiment: Experiment to append layer to.
        :param chip_coordinate: Chip coordinate to place to.
        """
        ExperimentElement.__init__(self, experiment)
        super().__init__(
            experiment,
            chip_coordinate=chip_coordinate,
        )
        self.size = size
        self.hw_observables = self._observables_factory()

    def _add_recorder_to_experiment(
            self,
            recorder: Union[gabstract.SpikeRecorder, gabstract.MADCRecorder,
                            gabstract.CADCRecorder],
            recorder_descriptor_name: str,
            recording_ids: grenade.common.ListMultiIndexSequence,
            source_port: int,
            experiment: ExperimentSnippet):

        recorder_vertex = recorder
        descriptor = getattr(self, recorder_descriptor_name)

        if descriptor is not None and \
                experiment.topology.contains(descriptor):
            experiment.topology.clear_vertex(descriptor)
            if recording_ids.size() == 0:
                experiment.topology.remove_vertex(descriptor)
                setattr(self, recorder_descriptor_name, None)
                descriptor = None
            else:
                experiment.topology.set(descriptor, recorder_vertex)
        elif recording_ids.size() != 0:
            descriptor = experiment.topology.add_vertex(recorder_vertex)
            setattr(self, recorder_descriptor_name, descriptor)
        if descriptor is not None:
            edge = grenade.common.Edge(
                recording_ids,
                grenade.common.CuboidMultiIndexSequence([recording_ids.size()]),
                source_port,
                0,
            )
            experiment.topology.add_edge(
                self.descriptor, descriptor, edge)

    def extra_repr(self) -> str:
        """ Add additional information """
        # TODO: move this to pytorch stuff
        reprs = f"experiment={self.experiment}, "
        reprs += f"{super().extra_repr()}"
        return reprs


class InputPopulation(BasePopulation):
    """ Base type for external input populations """
    log = logger.get("hxtorch.core.modules.InputPopulation")

    def __init__(
        self, size: int,
        experiment: BaseExperiment,
        chip_coordinate: Dict[
            grenade.common.ExecutionInstanceID,
            DLSGlobal
        ] | None = None,
        enable_spike_loopback: bool = False,
    ) -> None:
        """ """
        super().__init__(size, experiment, chip_coordinate)
        self._grenade_spike_loopback_descriptor = None
        self.enable_spike_loopback = enable_spike_loopback

    def add_to_topology(
        self,
        experiment: ExperimentSnippet,
    ):
        if self.descriptor is not None and \
                experiment.topology.contains(self.descriptor):
            experiment.topology.set(self.descriptor, self.generate_vertex())
        else:
            self.descriptor = experiment.topology.add_vertex(
                self.generate_vertex())
        self.log.TRACE(
            f"Added InputPopulation with descriptor: {self.descriptor}")

        # Add spike recorder
        if self.enable_spike_loopback:
            spike_recording_ids = grenade.common.ListMultiIndexSequence(
                [grenade.common.MultiIndex([pop_neuron_id, 0])
                 for pop_neuron_id in range(self.size)],
                [grenade.common.CellOnPopulationDimensionUnit(),
                 grenade.common.CompartmentOnNeuronDimensionUnit()],
            )
            spike_recorder = gabstract.SpikeRecorder(
                grenade.common.CuboidMultiIndexSequence([spike_recording_ids.size()]),
                grenade.common.TimeDomainOnTopology())
            self._add_recorder_to_experiment(
                spike_recorder, "_grenade_spike_loopback_descriptor",
                spike_recording_ids, 0, experiment)
            self.log.TRACE(
                "Added spike recorder with descriptor: ",
                self._grenade_spike_loopback_descriptor)

        return True

    def generate_vertex(self) \
            -> grenade.common.Population:
        return grenade.common.Population(
            gabstract.ExternalSourceNeuron(),
            grenade.common.CuboidMultiIndexSequence(
                [self.size],
                [grenade.common.CellOnPopulationDimensionUnit()]),
            gabstract.ExternalSourceNeuron.ParameterSpace(self.size),
            grenade.common.TimeDomainOnTopology())

    def add_to_parameterization(
            self, experiment: ExperimentSnippet):
        pass

    @abstractmethod
    def get_spike_times(self) -> List[List[List[grenade.common.Time]]]:
        """ """

    def generate_input_data(
        self,
        experiment: ExperimentSnippet,
        snippet_begin_time,
        snippet_end_time,
    ) -> Dict[int, grenade.common.PortData]:
        spike_times = self.get_spike_times()
        return {0: gabstract.ExternalSourceNeuron.Dynamics(spike_times)}

    def add_to_input_data(
        self,
        experiment: ExperimentSnippet,
        snippet_begin_time,
        snippet_end_time,
    ):
        input_data = self.generate_input_data(
            experiment,
            snippet_begin_time,
            snippet_end_time,
        )
        if input_data is None:
            return

        for port_on_vertex, port_data in input_data.items():
            experiment.input_data.ports.set(
                (self.descriptor, port_on_vertex), port_data)

    def extract_output_data(
        self,
        snippets: List[ExperimentSnippet],
    ):
        assert len(snippets) == 1
        spikes = None
        if self.enable_spike_loopback:
            spikes = snippets[0].output_data.ports.get(
                (self._grenade_spike_loopback_descriptor, 0)).spikes
            self.log.TRACE("Extracted spikes for InputPopulation: ", self)
        self.hw_observables.set_data(spikes=spikes)


class Population(BasePopulation):

    log = logger.get("hxtorch.core.modules.Population")

    _parameters_defaults = {  # pylint: disable=invalid-name
        "v_leak": 80,
        "v_reset": 80,
        "v_threshold": 125,
        "tau_membrane": 1e-5,
        "tau_syn_E": 1e-5,
        "tau_syn_I": 1e-5,
        "membrane_capacitance": 63,
        "i_synin_gm_E": 500,
        "i_synin_gm_I": 500,
        "tau_refrac": 2e-6,
        "synapse_dac_bias": 600,
        'e_rev_E': None,
        'e_rev_I': None,
        "leak_conductance": 1.,
        "exponential_slope": 50e-3,
        "exponential_threshold": 110,
        "subthreshold_adaptation_strength": 1,
        "leak_adaptation": None,
        "spike_triggered_adaptation_increment": 1,
        "tau_adap": 100e-6,
    }

    # Defines a mapping from parameter names used in the Population constructor
    # to the corresponding internal parameter names. This allows users to use
    # more intuitive parameter names when initializing a Population, while
    # maintaining compatibility with the internal parameter naming conventions
    # used, for instance, in the calix calibration targets.
    _param_name_mapping = {  # pylint: disable=invalid-name
        "v_leak": "leak",
        "v_reset": "reset",
        "v_threshold": "threshold",
        "tau_membrane": "tau_mem",
        "tau_syn_E": "tau_syn",
        "tau_syn_I": "tau_syn",
        "membrane_capacitance": "membrane_capacitance",
        "i_synin_gm_E": "i_synin_gm",
        "i_synin_gm_I": "i_synin_gm",
        "tau_refrac": "refractory_time",
        "synapse_dac_bias": "synapse_dac_bias",
        "e_rev_E": "e_rev_E",
        "e_rev_I": "e_rev_I",
        "leak_conductance": "leak_conductance",
        "exponential_slope": "exponential_slope",
        "exponential_threshold": "exponential_threshold",
        "subthreshold_adaptation_strength": "subthreshold_adaptation_strength",
        "leak_adaptation": "leak_adaptation",
        "spike_triggered_adaptation_increment": \
        "spike_triggered_adaptation_increment",
        "tau_adap": "tau_adap",
    }

    def __init__(
        self,
        size: int,
        experiment: BaseExperiment,
        leak: ModuleParameterType = 80,
        reset: ModuleParameterType = 80,
        threshold: ModuleParameterType = 125,
        tau_mem: ModuleParameterType = 10e-6,
        tau_syn: ModuleParameterType = 10e-6,
        i_synin_gm: ModuleParameterType = 500,
        membrane_capacitance: ModuleParameterType = 63,
        leak_conductance: Optional[HXBaseParameter] = None,
        refractory_time: ModuleParameterType = 1e-6,
        synapse_dac_bias: ModuleParameterType = 600,
        holdoff_time: ModuleParameterType = 0e-6,
        exponential_slope: ModuleParameterType = 50e-3,
        exponential_threshold: ModuleParameterType = 110,
        subthreshold_adaptation_strength: ModuleParameterType = 1,
        spike_triggered_adaptation_increment: ModuleParameterType = 1,
        clock_scale_adaptation_pulse: Tuple[int] = (5, 5),
        tau_adap: ModuleParameterType = 100e-6,
        leak_adaptation: Optional[HXBaseParameter] = None,
        chip_coordinate: Dict[
            grenade.common.ExecutionInstanceID,
            DLSGlobal
        ] | None = None,
        enable_spike_recording: bool = True,
        enable_cadc_recording: bool = True,
        enable_cadc_recording_placement_in_dram: bool = False,
        cadc_readout_source: hal.NeuronConfig.ReadoutSource = (
            ReadoutSource.VOLTAGE),
        enable_madc_recording: bool = False,
        enable_constant_current: bool = False,
        current_type: lola.AtomicNeuron.ConstantCurrent.Type = (
            lola.AtomicNeuron.ConstantCurrent.Type.source),
        record_neuron_id: int | None = None,
        madc_readout_source: hal.NeuronConfig.ReadoutSource = (
            ReadoutSource.VOLTAGE),
        placement_constraint: List[halco.LogicalNeuronOnDLS] | None = None,
        neuron_structure: Morphology | None = None,
        leaky: bool = True,
        fire: bool = True,
        exponential: bool = True,
        subthreshold_adaptation: bool = True,
        spike_triggered_adaptation: bool = True,
        **extra_params,
    ) -> None:
        """
        Initialize a Population. This module creates a population of neurons of
        size `size`. It includes an internal spiking mask, which allows
        disabling the event output and spike recordings of specific neurons
        within the layer. This is particularly useful for dropout.

        The neuron is parameterized by the `ModuleParameterType` parameters:
            leak, reset, threshold, tau_syn, tau_mem, i_synin_gm,
            membrane_capacitance, refractory_time, synapse_dac_bias,
            holdoff_time, exponential_slope, exponential_threshold, a, b,
            tau_adap.
        More information about these parameters on BSS-2 can be found in
        `calix.spiking.neuron.NeuronCalibTarget`. If the parameters are not
        provided as `ParameterType`, they are implicitly converted to
        `HXParameter`, which ensures the same value is used for the BSS-2
        calibration (`param.hardware_value`) and the numerical model
        (`param.model_value`) defined in `forward_func`.
        `MixedHXModelParameter` and `HXTransformedModelParameter` allow using
        different values on BSS-2 and in the numerical model. This is useful
        when the dynamic range on hardware and in the numerical model differ.
        In such cases, the trace and weight scaling parameters need to be set
        accordingly to translate the weights to their corresponding hardware
        value and the hardware measurements into the dynamic range used in the
        numerics.

        :param size: Size of the population.
        :param experiment: Experiment to append the layer to.
        :param leak: The leak potential. Defaults to HXParameter(80).
        :param reset: The reset potential. Defaults to HXParameter(80).
        :param threshold: The threshold potential. Defaults to
            HXParameter(125).
        :param tau_mem: The membrane time constant in seconds. Defaults to
            HXParameter(10e-6).
        :param tau_syn: The synaptic time constant in seconds. Defaults to
            HXParameter(10e-6).
        :param i_synin_gm: A hardware parameter adjusting the neuron-specific
            synaptic efficacy. Defaults to HXParameter(500).
        :param membrane_capacitance: The capacitance of the membrane. The
            available range is 0 to approximately 2.2 pF, represented as 0 to
            63 LSB.
        :param refractory_time: The refractory time constant in seconds.
            Defaults to HXParameter(1e-6).
        :param synapse_dac_bias: Synapse DAC bias current. Can be lowered to
            reduce the amplitude of a spike at the input of the synaptic input
            OTA. This can help avoid saturation when using larger synaptic time
            constants. Defaults to HXParameter(600).
        :param holdoff_time: Target length of the holdoff period in seconds.
            The holdoff period is the time at the end of the refractory period
            during which the clamping to the reset voltage is already released
            but new spikes cannot yet be generated. Defaults to
            HXParameter(0e-6).
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
        :param chip_coordinate: Chip coordinate this module is placed on.
        :param enable_spike_recording: Boolean flag to enable or disable spike
            recording. Note that this does not disable the event output of
            neurons. The event output must be disabled via `mask`.
        :param enable_cadc_recording: Enables or disables parallel sampling of
            the population's membrane trace via the CADC. A maximum sample rate
            of 1.7 µs is possible.
        :param enable_cadc_recording_placement_in_dram: Whether to place CADC
            recording data into DRAM (period ~6 µs) or SRAM (period ~2 µs).
        :param enable_madc_recording: Enables or disables the recording of the
            membrane trace of the neuron specified by `record_neuron_id` via
            the MADC. Only a single neuron can be recorded. This membrane trace
            is sampled with significantly higher resolution than with the CADC.
        :param enable_constant_current: Flag for enabling a constant on the
            neuron.
        :param current_type: Type of the current either source or sink.
            Defaults to source.
        :param record_neuron_id: The in-population neuron index of the neuron
            to be recorded with the MADC. This has an effect only when
            `enable_madc_recording` is enabled.
        :param placement_constraint: An optional list of logical neurons
            defining where to place the module's neurons on hardware.
        :param neuron_structure: Structure of the neuron. If not supplied, a
            single-compartment neuron circuit is used.
        :param leaky: Flag for enabling / disabling the leak term.
        :param fire: Flag for enabling / disabling the firing behaviour.
        :param exponential: Flag for enabling / disabling the exponential term.
        :param subthreshold_adaptation: Flag for enabling / disabling the
            subthreshold adaptation term.
        :param spike_triggered_adaptation: Flag for enabling / disabling the
            spike-triggered adaptation.
        """
        super().__init__(size, experiment, chip_coordinate)

        if placement_constraint is not None \
                and len(placement_constraint) != size:
            raise ValueError(
                "The number of neurons in logical neurons in "
                + "`hardware_constraints` does not equal the `size` of the "
                + "module.")

        self._cadc_readout_source = cadc_readout_source
        self._madc_readout_source = madc_readout_source

        self._enable_spike_recording = enable_spike_recording
        self._enable_cadc_recording = enable_cadc_recording
        self._enable_cadc_rec_in_dram = \
            enable_cadc_recording_placement_in_dram
        self._enable_madc_recording = enable_madc_recording
        self._record_neuron_id = record_neuron_id
        self.experiment.add_placement_constraint(size, placement_constraint)

        self.leaky = leaky
        self.fire = fire
        self.exponential = exponential
        self.subthreshold_adaptation = subthreshold_adaptation
        self.spike_triggered_adaptation = spike_triggered_adaptation
        self.clock_scale_adaptation_pulse = clock_scale_adaptation_pulse
        self.adaptation = self.subthreshold_adaptation \
            or self.spike_triggered_adaptation
        self._enable_constant_current = enable_constant_current
        self._current_type = current_type

        self._grenade_spike_descriptor = None
        self._grenade_madc_descriptor = None
        self._grenade_cadc_descriptor = None

        self._generate_hxparameters(
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
            subthreshold_adaptation_strength=subthreshold_adaptation_strength,
            leak_adaptation=leak_adaptation,
            spike_triggered_adaptation_increment=(
                spike_triggered_adaptation_increment
            ),
            tau_adap=tau_adap,
            **extra_params,
        )

        self._spike_mask = np.ones(self.size, dtype=bool)

        if neuron_structure is None:
            self._neuron_structure = SingleCompartmentNeuron(1)
        else:
            if len(neuron_structure.compartments.get_compartments()) > 1:
                # Issue #4020 (we always record the first neuron circuit in
                # the first compartment)
                raise ValueError('Currently only neurons with a single '
                                 'compartment are supported.')
            self._neuron_structure = neuron_structure

    def _generate_hxparameters(
        self, **hxparams: Dict[str, ModuleParameterType],
    ):
        # Set incoming parameters
        for param_key, param_value in hxparams.items():
            if not isinstance(param_value, HXBaseParameter):
                param_value = HXParameter(param_value)
            if param_key is not None:
                setattr(self, param_key, param_value)
        for param_key, param_value in \
                self._parameters_defaults.items():
            param_key_mapped = self._param_name_mapping.get(param_key)
            if hasattr(self, param_key):
                continue
            elif hasattr(self, param_key_mapped):
                setattr(self, param_key, getattr(self, param_key_mapped))
            else:
                setattr(self, param_key, HXParameter(param_value))

    def _trainable_parameters(self) -> List[HXBaseParameter]:
        params = {}
        for param_key in self._parameters_defaults.keys():
            if hasattr(getattr(self, param_key), "set_hw_config"):
                params[param_key] = getattr(self, param_key)
        return params

    def add_to_topology(
        self, experiment: ExperimentSnippet,
    ):
        # Add population
        if self.descriptor is not None and \
                experiment.topology.contains(self.descriptor):
            experiment.topology.set(
                self.descriptor, self.generate_vertex())
        else:
            self.descriptor = experiment.topology.add_vertex(
                self.generate_vertex())
        self.log.TRACE("Added population with descriptor: ", self.descriptor)

        # Add spike recorder
        if self._enable_spike_recording:
            spike_recording_ids = grenade.common.ListMultiIndexSequence(
                [grenade.common.MultiIndex(
                    [pop_neuron_id, CompartmentOnLogicalNeuron()]
                ) for pop_neuron_id in np.nonzero(self._spike_mask)[0]],
                [grenade.common.CellOnPopulationDimensionUnit(),
                 grenade.common.CompartmentOnNeuronDimensionUnit()])
            spike_recorder = gabstract.SpikeRecorder(
                grenade.common.CuboidMultiIndexSequence(
                    [spike_recording_ids.size()],
                ),
                grenade.common.TimeDomainOnTopology(),
            )
            self._add_recorder_to_experiment(
                spike_recorder,
                "_grenade_spike_descriptor",
                spike_recording_ids,
                0,
                experiment,
            )
            self.log.TRACE(
                "Added spike recorder with descriptor: ",
                self._grenade_spike_descriptor)

        # Add MADC recorder
        if self._enable_madc_recording:
            madc_recording_ids = grenade.common.ListMultiIndexSequence(
                [grenade.common.MultiIndex(  # twice for multiple
                    [self._record_neuron_id, CompartmentOnLogicalNeuron(), 0])],
                [grenade.common.CellOnPopulationDimensionUnit(),
                 grenade.common.CompartmentOnNeuronDimensionUnit(),
                 gabstract.AtomicNeuronOnCompartmentDimensionUnit()])
            madc_recorder = gabstract.MADCRecorder(
                grenade.common.CuboidMultiIndexSequence(
                    [madc_recording_ids.size()],
                ),
                grenade.common.TimeDomainOnTopology())
            self._add_recorder_to_experiment(
                madc_recorder,
                "_grenade_madc_descriptor",
                madc_recording_ids,
                1,
                experiment
            )
            self.log.TRACE(
                "Added MADC recorder with descriptor: ",
                self._grenade_madc_descriptor,
            )

        # Add CADC recorder
        if self._enable_cadc_recording:
            cadc_recording_ids = grenade.common.ListMultiIndexSequence(
                [grenade.common.MultiIndex(
                    [pop_neuron_id, CompartmentOnLogicalNeuron(), 0])
                 for pop_neuron_id in range(self.size)],
                [grenade.common.CellOnPopulationDimensionUnit(),
                 grenade.common.CompartmentOnNeuronDimensionUnit(),
                 gabstract.AtomicNeuronOnCompartmentDimensionUnit()])
            cadc_recorder = gabstract.CADCRecorder(
                grenade.common.CuboidMultiIndexSequence(
                    [cadc_recording_ids.size()],
                ),
                self._enable_cadc_rec_in_dram,
                grenade.common.TimeDomainOnTopology(),
            )
            self._add_recorder_to_experiment(
                cadc_recorder,
                "_grenade_cadc_descriptor",
                cadc_recording_ids,
                1,
                experiment
            )
            self.log.TRACE(
                "Added CADC recorder with descriptor: ",
                self._grenade_cadc_descriptor,
            )

        self.changed_topology = True

        return True

    def add_to_input_data(
        self,
        experiment: ExperimentSnippet,
        snippet_begin_time,
        snippet_end_time,
    ):
        input_data = self.generate_input_data(
            experiment,
            snippet_begin_time,
            snippet_end_time,
        )
        if input_data is None:
            return

        for port_on_vertex, port_data in input_data.items():
            experiment.input_data.ports.set(
                (self.descriptor, port_on_vertex),
                port_data,
            )

        self.changed_input_data = True

    def extract_output_data(
            self, snippets: List[ExperimentSnippet]):
        assert len(snippets) == 1
        spikes, cadc, madc = None, None, None
        if self._grenade_spike_descriptor is not None:
            spikes = snippets[0].output_data.ports.get(
                (self._grenade_spike_descriptor, 0),
            ).spikes
            self.log.TRACE(f"Extracted spikes for Population: {self}")
        if self._grenade_cadc_descriptor is not None:
            cadc = snippets[0].output_data.ports.get(
                (self._grenade_cadc_descriptor, 0),
            ).samples
            self.log.TRACE(f"Extracted CADC samples for Population: {self}")
        if self._grenade_madc_descriptor is not None:
            madc = snippets[0].output_data.ports.get(
                (self._grenade_madc_descriptor, 0),
            ).samples
            self.log.TRACE(f"Extracted MADC samples for Population: {self}")
        self.hw_observables.set_data(spikes=spikes, cadc=cadc, madc=madc)

    def generate_vertex(self) -> grenade.common.Population:
        ans = self._neuron_structure.logical_neuron.collapse_neuron()
        # only one compartment is supported
        assert len(ans) == 1
        n_ans = len(ans[CompartmentOnLogicalNeuron()])

        calibration_targets = self.generate_calibration_targets()

        membrane_capacitances = [{
            grenade.common.CompartmentOnNeuron():
            CapacitanceInFarad(
                grenade_vx.ideal_capacitance_per_neuron * partial(
                    self.resize_parameter_value, i, self.size)(
                    self.membrane_capacitance)
                / hal.NeuronConfig.MembraneCapacitorSize.max)}
            for i in range(self.size)]

        parameter_space = gabstract.CalibratedNeuron.ParameterSpace(
            calibration_targets, membrane_capacitances)

        return grenade.common.Population(
            gabstract.CalibratedNeuron(
                {grenade.common.CompartmentOnNeuron():
                 gabstract.CalibratedNeuron.Compartment(
                     # TODO: Enable / disable spiking
                     gabstract.CalibratedNeuron.Compartment.SpikeMaster(0),
                     # convention
                     [{grenade.common.ReceptorOnCompartment(0):
                       grenade.network.Receptor.Type.excitatory,
                       grenade.common.ReceptorOnCompartment(1):
                       grenade.network.Receptor.Type.inhibitory}] * n_ans)},
                self._neuron_structure.compartments),
            grenade.common.CuboidMultiIndexSequence(
                [self.size],
                [grenade.common.CellOnPopulationDimensionUnit()]),
            parameter_space,
            grenade.common.TimeDomainOnTopology())

    @staticmethod
    def resize_parameter_value(neuron_id, size, param):
        if param is None:
            return None
        val = param.hardware_value
        if isinstance(val, np.ndarray) and val.size == size:
            return val[neuron_id]
        elif isinstance(val, np.ndarray) and val.size == 1:
            return val.item()
        return val

    def generate_calibration_targets(self) \
            -> List[Dict[grenade.common.CompartmentOnNeuron,
                         List[gabstract.CalibratedNeuron
                              .ParameterSpace.CalibrationTarget]]]:
        ans = self._neuron_structure.logical_neuron.collapse_neuron()
        # only one compartment is supported
        assert len(ans) == 1
        n_ans = len(ans[CompartmentOnLogicalNeuron()])
        calibration_targets = []

        for neuron_id in range(self.size):
            calibration_target = gabstract.CalibratedNeuron.ParameterSpace\
                .CalibrationTarget()

            calibration_target.synaptic_input_excitatory = \
                gabstract.CalibratedNeuron.ParameterSpace\
                .CalibrationTarget.CubaSynapticInput()
            calibration_target.synaptic_input_inhibitory = \
                gabstract.CalibratedNeuron.ParameterSpace\
                .CalibrationTarget.CubaSynapticInput()
            calibration_target.refractory_period = \
                gabstract.CalibratedNeuron.ParameterSpace\
                .CalibrationTarget.RefractoryPeriod()

            get_val = partial(
                self.resize_parameter_value,
                neuron_id,
                self.size,
            )

            calibration_target.membrane_capacitance_during_calibration = \
                CapacitanceInFarad(
                    grenade_vx.ideal_capacitance_per_neuron * get_val(
                        self.membrane_capacitance)
                    / hal.NeuronConfig.MembraneCapacitorSize.max)
            calibration_target.v_leak = int(get_val(self.v_leak))
            calibration_target.tau_membrane = TimeInS(
                float(get_val(self.tau_mem))
            )
            calibration_target.v_threshold = int(get_val(self.v_threshold))
            calibration_target.v_reset = int(get_val(self.v_reset))
            calibration_target.refractory_period.refractory_time = \
                TimeInS(float(get_val(self.tau_refrac)))
            calibration_target.synaptic_input_excitatory.i_synin_gm = \
                int(get_val(self.i_synin_gm_E))
            calibration_target.synaptic_input_inhibitory.i_synin_gm = \
                int(get_val(self.i_synin_gm_I))
            calibration_target.synaptic_input_excitatory.tau_syn = \
                TimeInS(float(get_val(self.tau_syn_E)))
            calibration_target.synaptic_input_inhibitory.tau_syn = \
                TimeInS(float(get_val(self.tau_syn_I)))
            calibration_target.synaptic_input_excitatory.synapse_dac_bias = \
                int(get_val(self.synapse_dac_bias))
            calibration_target.synaptic_input_inhibitory.synapse_dac_bias = \
                int(get_val(self.synapse_dac_bias))

            e_rev_e_val = get_val(self.e_rev_E)
            if e_rev_e_val is not None:
                calibration_target.synaptic_input_excitatory.e_reversal = \
                    int(e_rev_e_val)

            e_rev_i_val = get_val(self.e_rev_I)
            if e_rev_i_val is not None:
                calibration_target.synaptic_input_inhibitory.e_reversal = \
                    int(e_rev_i_val)

            # We assume same parameters for all atomic neurons in the
            # compartment
            calibration_targets.append({
                grenade.common.CompartmentOnNeuron(): n_ans * [calibration_target]
            })

        return calibration_targets

    def override_hw_params(
        self,
        experiment: ExperimentSnippet,
    ) -> Dict[int, grenade.common.PortData]:
        """ Change hardware parameters. This does not trigger calibration """
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
                partitioned_vertex_descriptors = experiment.mapped_topology\
                    .get_reference().links(
                        compatible_inter_graph_hyper_edge_descriptor)
                assert len(partitioned_vertex_descriptors) == 1
                section = experiment.mapped_topology.get_reference().get(
                    partitioned_vertex_descriptors[0])

                for section_mapping_descriptor in experiment.mapped_topology\
                        .inter_graph_hyper_edges_by_reference(
                            partitioned_vertex_descriptors[0]):
                    section_mapping = experiment.mapped_topology.get(
                        section_mapping_descriptor)
                    if isinstance(
                            section_mapping,
                            grenade.network.abstract.ChipMapping):
                        continue
                    coords = []
                    configs = []
                    for i, element in enumerate(
                            section.get_shape().get_elements()):
                        anchor = section_mapping.anchors[i]
                        coord = halco.LogicalNeuronOnDLS(
                            self._neuron_structure.compartments, anchor[1],
                        )
                        comp = grenade.common.CompartmentOnNeuron()
                        nrn_configs = {comp: []}
                        for n in range(
                            len(list(self._neuron_structure.logical_neuron
                                .collapse_neuron().values())[0])):
                            an_config = section_mapping.get_config(i, comp, n)
                            nrn_configs[comp].append(an_config)
                        coords.append(coord)
                        configs.append(nrn_configs)
                    # TODO: What about in-pop neuron order?
                    #       Might require more than config
                    for param in self._trainable_parameters().values():
                        param.set_hw_config(coords, configs)

    def generate_input_data(
        self,
        experiment: ExperimentSnippet,
        snippet_begin_time,
        snippet_end_time,
    ) -> Dict[int, grenade.common.PortData]:

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
                partitioned_vertex_descriptors = experiment.mapped_topology\
                    .get_reference().links(
                        compatible_inter_graph_hyper_edge_descriptor)
                assert len(partitioned_vertex_descriptors) == 1
                section = experiment.mapped_topology.get_reference().get(
                    partitioned_vertex_descriptors[0])

                chip_mapping = None
                for section_mapping_descriptor in experiment.mapped_topology\
                        .inter_graph_hyper_edges_by_reference(
                            partitioned_vertex_descriptors[0]):
                    section_mapping = experiment.mapped_topology.get(
                        section_mapping_descriptor)
                    if isinstance(
                            section_mapping,
                            grenade.network.abstract.ChipMapping):
                        chip_mapping = section_mapping
                        continue
                    backends = []
                    if chip_mapping is not None \
                            and chip_mapping.base is not None:
                        backends = [
                            chip_mapping.base.neuron_block.backends[coord]
                            for coord in halco.iter_all(
                                halco.CommonNeuronBackendConfigOnDLS)
                        ]
                    for i, element in enumerate(
                            section.get_shape().get_elements()):
                        anchor = section_mapping.anchors[i]
                        coord = halco.LogicalNeuronOnDLS(
                            self._neuron_structure.compartments, anchor[1],
                        )
                        comp = grenade.common.CompartmentOnNeuron()
                        configs = {comp: []}
                        for n in range(
                            len(list(self._neuron_structure.logical_neuron
                                .collapse_neuron().values())[0])):
                            an_config = section_mapping.get_config(i, comp, n)
                            configs[comp].append(an_config)
                        self._configure_neuron_structure(
                            coord,
                            configs,
                            backends,
                            element,
                        )

        # readout source defaulting to membrane
        ans = self._neuron_structure.logical_neuron.collapse_neuron()
        # only one compartment is supported
        assert len(ans) == 1
        n_ans = len(ans[CompartmentOnLogicalNeuron()])
        readout_sources = [
            {
                grenade.common.CompartmentOnNeuron():
                [self._cadc_readout_source] * n_ans
            }
            for i in range(self.size)
        ]

        calibration_targets = self.generate_calibration_targets()

        membrane_capacitances = [{
            grenade.common.CompartmentOnNeuron():
            CapacitanceInFarad(
                grenade_vx.ideal_capacitance_per_neuron * partial(
                    self.resize_parameter_value, i, self.size)(
                    self.membrane_capacitance)
                / hal.NeuronConfig.MembraneCapacitorSize.max)}
            for i in range(self.size)]

        return {
            1: gabstract.CalibratedNeuron.ParameterSpace.Parameterization(
                calibration_targets,
                membrane_capacitances,
                readout_sources,
            )
        }

    def _configure_neuron_structure(
        self,
        coord: halco.LogicalNeuronOnDLS,
        configs: Dict[grenade.common.CompartmentOnNeuron, List[int]],
        backends: List[hal.CommonNeuronBackend],
        element: grenade.common.Element,
    ):
        neuron_id = element.value[0]

        # Set all parameters of the exponential term and the adaptation term.
        # Get user-defined hardware parameters
        get_val = partial(
            self.resize_parameter_value,
            neuron_id,
            self.size,
        )

        exponential_threshold = get_val(self.exponential_threshold)
        exponential_slope = get_val(self.exponential_slope)
        tau_adap = get_val(self.tau_adap)
        subthreshold_adaptation_strength = get_val(
            self.subthreshold_adaptation_strength
        )
        leak_adaptation = get_val(self.leak_adaptation)
        spike_triggered_adaptation_increment = get_val(
            self.spike_triggered_adaptation_increment
        )

        # Configure neuron structure
        self._neuron_structure.implement_morphology(
            coord, configs
        )
        if not self.leaky:
            self._neuron_structure.disable_leak(configs)
        if self.fire:
            self._neuron_structure.set_spike_recording(
                self._spike_mask[neuron_id],
                configs,
            )
        else:
            self._neuron_structure.disable_spiking(configs)
        if self._enable_madc_recording and (
                neuron_id == self._record_neuron_id):
            self._neuron_structure.enable_madc_recording(
                configs,
                self._madc_readout_source,
            )
        if self._enable_constant_current:
            assert self._current_type is not None
            self._neuron_structure.enable_constant_current(
                configs,
                self._current_type,
            )
        if self.exponential:
            self._neuron_structure.set_exponential_params(
                configs,
                exponential_threshold,
                exponential_slope,
            )
        if self.adaptation:
            self._neuron_structure.set_adaptation_base_params(
                configs,
                tau_adap,
            )
        if self.subthreshold_adaptation:
            self._neuron_structure.set_subthreshold_adaptation_strength(
                configs,
                subthreshold_adaptation_strength,
                leak_adaptation,
            )
        if self.spike_triggered_adaptation:
            self._neuron_structure.set_spike_triggered_adaptation_increment(
                configs,
                backends,
                spike_triggered_adaptation_increment,
                self.clock_scale_adaptation_pulse,
            )

    def params_dict(self) -> Dict:
        return {param: getattr(self, param, None)
                for param in self._parameters_defaults}
