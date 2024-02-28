"""
Defining basic types to create hw-executable instances
"""
# pylint: disable=no-member, invalid-name
from typing import Callable, Dict, List, Tuple, Union, Optional
from abc import ABC, abstractmethod
from pathlib import Path
import itertools
import pylogging as logger

import torch
import numpy as np

from dlens_vx_v3 import hal, sta, lola
import pygrenade_vx as grenade

import _hxtorch_spiking  # pylint: disable=no-name-in-module
from hxtorch.spiking import modules as spiking_modules
from hxtorch.spiking import handle
from hxtorch.spiking.utils import calib_helper
from hxtorch.spiking.neuron_placement import NeuronPlacement
from hxtorch.spiking.backend.module_manager import (
    BaseModuleManager, ModuleManager)

log = logger.get("hxtorch.spiking.experiment")


class BaseExperiment(ABC):

    def __init__(self, modules: BaseModuleManager, mock: bool, dt: float) \
            -> None:
        self.mock = mock
        self.modules = modules
        self.dt = dt

    @abstractmethod
    def connect(self, module: torch.nn.Module,
                input_handles: handle.TensorHandle,
                output_handle: handle.TensorHandle):
        raise NotImplementedError

    @abstractmethod
    def get_hw_results(self, runtime: Optional[int]) \
            -> Dict[grenade.network.PopulationOnNetwork,
                    Tuple[Optional[torch.Tensor], ...]]:
        raise NotImplementedError


class Experiment(BaseExperiment):

    """ Experiment class for describing experiments on hardware """

    # pylint: disable=too-many-arguments
    def __init__(
            self, mock: bool = False, dt: float = 1e-6,
            calib_path: Optional[Union[Path, str]] = None,
            hw_routing_func=grenade.network.routing.PortfolioRouter()) -> None:
        """
        Instantiate a new experiment, representing an experiment on hardware
        and/or in software.

        :param mock: Indicating whether module is executed on hardware (False)
            or simulated in software (True).
        """
        super().__init__(ModuleManager(), mock=mock, dt=dt)

        # Load chip object
        self._chip = None
        if calib_path is not None:
            self._chip = self.load_calib(calib_path)

        # Recording
        self.cadc_recording = {}
        self.has_madc_recording = False

        # Grenade stuff
        self.grenade_network = None
        self.grenade_network_graph = None

        # Configs
        self._static_config_prepared = False
        self.injection_pre_static_config = None
        self.injection_pre_realtime = None  # Unused
        self.injection_post_realtime = None  # Unused
        self.injection_inside_realtime_begin = None  # Unused
        self.injection_inside_realtime = None  # Unused
        self.injection_inside_realtime_end = None  # Unused

        self._populations: List[spiking_modules.HXModule] = []
        self._projections: List[spiking_modules.HXModule] = []

        self._batch_size = 0
        self.id_counter: Dict[
            grenade.common.ExecutionInstanceID, int] = {}

        self.neuron_placement: Dict[
            grenade.common.ExecutionInstanceID, NeuronPlacement] = {}
        self.hw_routing_func = hw_routing_func

        self.inter_batch_entry_wait = None

        # Last run results
        self._last_run_chip_configs = None

    def clear(self) -> None:
        """
        Reset the experiments's state. Corresponds to creating a new Experiment
        instance.
        """
        self.modules.clear()

        self.cadc_recording = {}
        self.has_madc_recording = False

        self.grenade_network = None
        self.grenade_network_graph = None

        self._static_config_prepared = False
        self.injection_pre_static_config = None
        self.injection_pre_realtime = None  # Unused
        self.injection_post_realtime = None  # Unused

        self._populations = []
        self._projections = []

        self._batch_size = 0
        self.id_counter = {}

        self.inter_batch_entry_wait = None

    def _prepare_static_config(self) -> None:
        """
        Prepares all the static chip config. Accesses the chip object
        initialized by hxtorch.hardware_init and appends corresponding
        configurations to. Additionally this method defines the
        pre_static_config builder injected to grenade at run.
        """
        if self._static_config_prepared:  # Only do this once
            return

        # If chip is still None we load default nightly calib
        if self._chip is None:
            log.INFO(
                "No chip object present. Using chip object with default "
                + "nightly calib.")
            self._chip = self.load_calib(calib_helper.nightly_calib_path())

        # NOTE: Reserved for inserted config.
        builder = sta.PlaybackProgramBuilder()
        pre_realtime = sta.PlaybackProgramBuilder()
        inside_realtime_begin = sta.PlaybackProgramBuilder()
        inside_realtime = sta.AbsoluteTimePlaybackProgramBuilder()
        inside_realtime_end = sta.PlaybackProgramBuilder()
        post_realtime = sta.PlaybackProgramBuilder()

        self.injection_pre_static_config = builder
        self.injection_pre_realtime = pre_realtime
        self.injection_inside_realtime_begin = inside_realtime_begin
        self.injection_inside_realtime = inside_realtime
        self.injection_inside_realtime_end = inside_realtime_end
        self.injection_post_realtime = post_realtime

        self._static_config_prepared = True
        log.TRACE("Preparation of static config done.")

    def load_calib(self, calib_path: Optional[Union[Path, str]] = None) \
            -> lola.Chip:
        """
        Load a calibration from path `calib_path` and apply to the experiment`s
        chip object. If no path is specified a nightly calib is applied.
        :param calib_path: The path to the calibration. It None, the nightly
            calib is loaded.
        :return: Returns the chip object for the given calibration.
        """
        # If no calib path is given we load spiking nightly calib
        log.INFO(f"Loading calibration from {calib_path}")
        self._chip = calib_helper.chip_from_file(calib_path)
        return self._chip

    def _generate_network_graphs(self) -> grenade.network.NetworkGraph:
        """
        Generate grenade network graph from the populations and projections in
        modules

        :return: Returns the grenade network graph.
        """
        changed_since_last_run = self.modules.changed_since_last_run()

        log.TRACE(f"Network changed since last run: {changed_since_last_run}")
        if not changed_since_last_run:
            if self.grenade_network_graph is not None:
                return self.grenade_network_graph

        # Create network builder
        network_builder = grenade.network.NetworkBuilder()

        # Add populations
        for module in self._populations:
            module.descriptor = module.add_to_network_graph(network_builder)
        # Add projections
        for module in self._projections:
            pre_pop = self.modules.source_populations(module)
            post_pop = self.modules.target_populations(module)
            assert len(pre_pop) == 1, "On hardware, a projection can only " \
                "have one source population."
            assert len(post_pop) == 1, "On hardware, a projection can only " \
                "have one target population."
            module.descriptor = module.add_to_network_graph(
                pre_pop.pop().descriptor, post_pop.pop().descriptor,
                network_builder)

        # Add CADC recording
        if self.cadc_recording:
            for execution_instance, neurons in self.cadc_recording.items():
                cadc_recording = grenade.network.CADCRecording()
                cadc_recording.neurons = [v[0] for _, v in neurons.items()]
                network_builder.add(cadc_recording, execution_instance)

        network = network_builder.done()

        # route network if required
        routing_result = None
        if self.grenade_network_graph is None \
                or grenade.network.requires_routing(
                    network, self.grenade_network_graph):
            routing_result = self.hw_routing_func(network)

        # Keep graph
        self.grenade_network = network

        # build or update network graph
        if routing_result is not None:
            self.grenade_network_graph = grenade.network\
                .build_network_graph(
                    self.grenade_network, routing_result)
        else:
            grenade.network.update_network_graph(
                self.grenade_network_graph,
                self.grenade_network)

        return self.grenade_network_graph

    def _configure_populations(self):
        """
        Configure the population on hardware.
        """
        # Make sure experiment holds chip config
        assert self._chip is not None

        pop_changed_since_last_run = any(
            m.changed_since_last_run for m in self._populations)
        if not pop_changed_since_last_run:
            return
        for module in self._populations:
            if not isinstance(module, spiking_modules.Neuron):
                continue
            log.TRACE(f"Configure population '{module}'.")
            for in_pop_id, unit_id in enumerate(module.unit_ids):
                coord = self.neuron_placement[
                    module.execution_instance
                ].id2logicalneuron(unit_id)
                self._chip.neuron_block = module.configure_hw_entity(
                    in_pop_id, self._chip.neuron_block, coord)
                log.TRACE(
                    f"Configured neuron at coord {coord}.")

    def _generate_inputs(
        self, network_graph: grenade.network.NetworkGraph) \
            -> grenade.signal_flow.InputData:
        """
        Generate external input events from the routed network graph
        representation.
        """
        # Make sure all batch sizes are equal
        sizes = [
            handle.observable_state.shape[1] for handle in
            self.modules.input_data()]
        assert all(sizes)
        self._batch_size = sizes[0]

        input_generator = grenade.network.InputGenerator(
            network_graph, self._batch_size)
        for module in self._populations:
            in_handle = [
                e["handle"] for _, _, e in self.modules.graph.in_edges(
                    self.modules.get_id_by_module(module), data=True)].pop()
            module.add_to_input_generator(in_handle, input_generator)

        return input_generator.done()

    def _generate_hooks(self) \
            -> grenade.signal_flow.ExecutionInstanceHooks:
        """ Handle injected config (not supported yet) """
        assert self.injection_pre_static_config is not None
        assert self.injection_pre_realtime is not None
        assert self.injection_inside_realtime_begin is not None
        assert self.injection_inside_realtime is not None
        assert self.injection_inside_realtime_end is not None
        assert self.injection_post_realtime is not None
        pre_static_config = sta.PlaybackProgramBuilder()
        pre_realtime = sta.PlaybackProgramBuilder()
        inside_realtime_begin = sta.PlaybackProgramBuilder()
        inside_realtime = sta.AbsoluteTimePlaybackProgramBuilder()
        inside_realtime_end = sta.PlaybackProgramBuilder()
        post_realtime = sta.PlaybackProgramBuilder()
        pre_static_config.copy_back(self.injection_pre_static_config)
        pre_realtime.copy_back(self.injection_pre_realtime)
        inside_realtime_begin.copy_back(self.injection_inside_realtime_begin)
        inside_realtime.copy(self.injection_inside_realtime)
        inside_realtime_end.copy_back(self.injection_inside_realtime_end)
        post_realtime.copy_back(self.injection_post_realtime)
        return grenade.signal_flow.ExecutionInstanceHooks(
            pre_static_config, pre_realtime, inside_realtime_begin,
            inside_realtime, inside_realtime_end, post_realtime)

    def _get_population_observables(
            self, network_graph: grenade.network.NetworkGraph,
            result_map: grenade.signal_flow.OutputData, runtime) -> Dict[
                grenade.network.PopulationOnNetwork,
                np.ndarray]:
        """
        Takes the grenade network graph and the result map returned by grenade
        after experiment execution and returns a data map where for each
        population descriptor of registered populations the population-specific
        hardware observables are represented as Optional[torch.Tensor]s.
        Note: This function calls the modules `post_process` method.
        :param network_graph: The logical grenade network graph describing the
            logic of th experiment.
        :param result_map: The result map returned by grenade holding all
            recorded hardware observables.
        :param runtime: The runtime of the experiment executed on hardware in
            ms.
        :returns: Returns the data map as dict, where the keys are the
            population descriptors and values are tuples of values returned by
            the corresponding module's `post_process` method.
        """
        # Get hw data
        hw_spike_times = _hxtorch_spiking.extract_spikes(
            result_map, network_graph)
        hw_cadc_samples = _hxtorch_spiking.extract_cadc(
            result_map, network_graph)
        hw_madc_samples = _hxtorch_spiking.extract_madc(
            result_map, network_graph)

        # Data maps
        data_map: Dict[
            grenade.network.PopulationsDescriptor,
            Tuple[torch.Tensor]] = {}  # pylint: disable=c-extension-no-member

        # Map populations to data
        for module in self._populations:
            if isinstance(module, spiking_modules.InputNeuron):
                continue
            data_map[module.descriptor] = module.post_process(
                hw_spike_times.get(module.descriptor),
                hw_cadc_samples.get(module.descriptor),
                hw_madc_samples.get(module.descriptor),
                runtime / int(hal.Timer.Value.fpga_clock_cycles_per_us) / 1e6)

        return data_map

    def connect(self, module: torch.nn.Module,
                input_handles: Tuple[handle.TensorHandle],
                output_handle: handle.TensorHandle):
        """
        Add an module to the experiment and connect it to other experiment
        modules via input and output handles.

        :param module: The HXModule to add to the experiment.
        :param input_handles: The TensorHandle serving as input to the module
            (its obsv_state).
        :param output_handle: The TensorHandle outputted by the module,
            serving as input to subsequent HXModules.
        """
        return self.modules.add_node(
            module, input_handles, output_handle)

    def wrap_modules(self, modules: List[spiking_modules.HXModule],
                     func: Optional[Callable] = None):
        """
        Wrap a number of given modules into a wrapper to which a single
        function `func` can be assigned. In the PyTorch graph the individual
        module functions are then bypassed and only the wrapper's function is
        considered when building the PyTorch graph. This functionality is of
        interest if several modules have cyclic dependencies and need to be
        represented by one PyTorch function.
        :param modules: A list of module to be wrapped. These modules need to
            constitute a closed sub-graph with no modules in between that are
            not element of the wrapper.
        :func: The function to assign to the wrapper.
            TODO: Add info about this functions signature.
        """
        # Unique modules
        assert len(set(modules)) == len(modules)

        # Check if modules are already existent
        for wrapper in self.modules.wrappers:
            if set(wrapper.modules) == set(modules):
                wrapper.update(modules, func)
                return
            if wrapper.contains(modules):
                raise ValueError(
                    "You tried to register a group of modules that are "
                    + "partially registered in another group")

        self.modules.add_wrapper(
            spiking_modules.HXModuleWrapper(self, modules, func))

    def register_population(self, module: spiking_modules.HXModule) -> None:
        """
        Register a module as population.

        :param module: The module to register as population.
        """
        self._populations.append(module)

    def register_projection(self, module: spiking_modules.HXModule) -> None:
        """
        Register a module as projection.

        :param module: The module to register as projection.
        """
        self._projections.append(module)

    def get_hw_results(self, runtime: Optional[int]) \
            -> Dict[grenade.network.PopulationOnNetwork,
                    Tuple[Optional[torch.Tensor], ...]]:
        """
        Executes the experiment in mock or on hardware using the information
        added to the experiment for a time given by `runtime` and returns a
        dict of hardware data represented as PyTorch data types.

        :param runtime: The runtime of the experiment on hardware in ms.

        :returns: Returns the data map as dict, where the keys are the
            population descriptors and values are tuples of values returned by
            the corresponding module's `post_process` method.
        """
        if not self.mock:
            self._prepare_static_config()

        # Preprocess layer
        self.modules.pre_process(self)

        # In mock-mode nothing to do here
        if self.mock:
            return {}, None

        # Register HW entity
        for module in self.modules.nodes:
            if hasattr(module, "register_hw_entity") and module \
                    not in itertools.chain(self._projections,
                                           self._populations):
                module.register_hw_entity()

        # Generate network graph
        network = self._generate_network_graphs()

        # configure populations
        self._configure_populations()

        # handle runtim
        runtime_in_clocks = int(
            runtime * self.dt * 1e6
            * int(hal.Timer.Value.fpga_clock_cycles_per_us))
        if runtime_in_clocks > hal.Timer.Value.max:
            max_runtime = hal.Timer.Value.max /\
                int(hal.Timer.Value.fpga_clock_cycles_per_us)
            raise ValueError(
                f"Runtime of {runtime} to long. Maximum supported runtime "
                + f"{max_runtime}")

        # generate external spike trains
        inputs = self._generate_inputs(network)
        inputs.runtime = [{
            execution_instance: runtime_in_clocks for execution_instance
            in network.network.topologically_sorted_execution_instance_ids
        }] * self._batch_size
        log.TRACE(f"Registered runtimes: {inputs.runtime}")

        if self.inter_batch_entry_wait is not None:
            inputs.inter_batch_entry_wait = {
                execution_instance: self.inter_batch_entry_wait
                for execution_instance in network.network.
                topologically_sorted_execution_instance_ids
            }

        chips = {execution_instance: self._chip
                 for execution_instance in self.id_counter}

        hooks = {
            execution_instance: self._generate_hooks()
            for execution_instance in self.id_counter}

        outputs = _hxtorch_spiking.run(
            chips, network, inputs, hooks)

        hw_data = self._get_population_observables(
            network, outputs, runtime_in_clocks)

        self.modules.reset_changed_since_last_run()

        self._last_run_chip_configs = outputs.pre_execution_chips

        return hw_data, outputs.execution_time_info

    @property
    def last_run_chip_configs(self) -> grenade.signal_flow.OutputData:
        return self._last_run_chip_configs
