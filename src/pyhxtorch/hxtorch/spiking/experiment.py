"""
Defining basic types to create hw-executable instances
"""
# pylint: disable=no-member, invalid-name
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import itertools
import pylogging as logger

import torch
import numpy as np

from dlens_vx_v3 import hal
import pygrenade_vx as grenade

import _hxtorch_spiking  # pylint: disable=no-name-in-module
from hxtorch.spiking.observables import HardwareObservablesExtractor
from hxtorch.spiking.execution_instance import (
    ExecutionInstances, ExecutionInstance)
from hxtorch.spiking import modules as spiking_modules
from hxtorch.spiking import handle
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
            hw_routing_func=grenade.network.routing.PortfolioRouter()) -> None:
        """
        Instantiate a new experiment, representing an experiment on hardware
        and/or in software.

        :param mock: Indicating whether module is executed on hardware (False)
            or simulated in software (True).
        :param input_loopback: Record input spikes and use them for gradient
            calculation. Depending on link congestion, this may or may not be
            beneficial for the calculated gradient's precision.
        """
        super().__init__(ModuleManager(), mock=mock, dt=dt)

        self._execution_instances = ExecutionInstances()

        self.hw_routing_func = hw_routing_func

        # Grenade stuff
        self.grenade_network = None
        self.grenade_network_graph = None

        # Configs
        self._static_config_prepared = False
        self._default_execution_instance: Optional[ExecutionInstance] = None

        self._populations: List[spiking_modules.HXModule] = []
        self._projections: List[spiking_modules.HXModule] = []

        self._hw_data_extractor = HardwareObservablesExtractor()
        self._batch_size = 0
        self.inter_batch_entry_wait = None

        # Last run results
        self._last_run_chip_configs = None

    def clear(self) -> None:
        """
        Reset the experiments's state. Corresponds to creating a new Experiment
        instance.
        """
        self.modules.clear()
        self._execution_instances.clear()

        self.grenade_network = None
        self.grenade_network_graph = None

        self.inter_batch_entry_wait = None
        self._static_config_prepared = False
        self._default_execution_instance = None

        self._populations = []
        self._projections = []

        self._batch_size = 0

    @property
    def default_execution_instance(self) -> ExecutionInstance:
        """
        Getter for the default ``ExecutionInstance`` object. All modules that
        have the same ``Experiment`` instance assigned and do not hold an
        explicit ``ExecutionInstance`` are assigned to this default execution
        instance.

        :return: The default execution instance
        """
        if self._default_execution_instance is None:
            self._default_execution_instance = ExecutionInstance()
        return self._default_execution_instance

    @default_execution_instance.setter
    def default_execution_instance(
            self, execution_instance: Optional[ExecutionInstance]) -> None:
        """
        Setter for the default ``ExecutionInstance`` object. All modules that
        have the same ``Experiment`` instance assigned and do not hold an
        explicit ``ExecutionInstance`` are assigned to this default execution
        instance.

        :param: The default execution instance to be used
        """
        self._default_execution_instance = execution_instance

    def _prepare_static_config(self) -> None:
        """
        Prepares all the static chip config. Accesses the chip object
        initialized by hxtorch.hardware_init and appends corresponding
        configurations to. Additionally this method defines the
        pre_static_config builder injected to grenade at run.
        """
        self._execution_instances.update([
            module.execution_instance for module in self.modules.nodes])
        if self._static_config_prepared:  # Only do this once
            return
        for execution_instance in self._execution_instances:
            modules = [m for m in self.modules.nodes
                       if m.execution_instance == execution_instance]
            execution_instance.modules = modules
        self._static_config_prepared = True
        log.TRACE("Preparation of static config done.")

    def _generate_network_graphs(self) -> grenade.network.NetworkGraph:
        """
        Generate grenade network graph from the populations and projections in
        modules

        TODO: Make this more ExecutionInstance specific

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
        for execution_instance, cadc_recording in self._execution_instances \
                .cadc_recordings.items():
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
            self.grenade_network_graph = grenade.network \
                .build_network_graph(self.grenade_network, routing_result)
        else:
            grenade.network.update_network_graph(
                self.grenade_network_graph, self.grenade_network)

        return self.grenade_network_graph

    def _configure_populations(self):
        """
        Configure the population on hardware.
        """
        pop_changed_since_last_run = any(
            m.changed_since_last_run for m in self._populations)
        if not pop_changed_since_last_run:
            return

        for module in self._populations:
            if not isinstance(module, spiking_modules.Neuron):
                continue
            log.TRACE(f"Configure population '{module}'.")
            for in_pop_id, unit_id in enumerate(module.unit_ids):
                coord = module.execution_instance.neuron_placement \
                    .id2logicalneuron(unit_id)
                module.execution_instance.chip.neuron_block = \
                    module.configure_hw_entity(
                        in_pop_id, module.execution_instance.chip.neuron_block,
                        coord)
                log.TRACE(f"Configured neuron at coord {coord}.")

    def _generate_inputs(
        self, network_graph: grenade.network.NetworkGraph) \
            -> grenade.signal_flow.InputData:
        """
        Generate external input events from the routed network graph
        representation.
        """
        # Make sure all batch sizes are equal
        sizes = [
            handle.spikes.shape[1] for handle in
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

    def _get_observables(
            self, network_graph: grenade.network.NetworkGraph,
            result_map: grenade.signal_flow.OutputData, runtime) -> Dict[
                grenade.network.PopulationOnNetwork,
                np.ndarray]:
        """
        Takes the grenade network graph and the result map returned by grenade
        after experiment execution and returns a data map where for each
        module descriptor of a registered module the specific hardware
        observables are represented as Optional[torch.Tensor]s.

        ..note: This function calls the modules `post_process` method.

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
        self._hw_data_extractor.set_data(network_graph, result_map)

        # Data maps
        data_map: Dict[
            grenade.network.PopulationsDescriptor,
            Tuple[torch.Tensor, ...]] = {}  # pylint: disable=c-extension-no-member

        # Map populations to data
        for module in self.modules.nodes:
            # Consider only hardware module
            if not isinstance(module, spiking_modules.HXModule):
                continue
            data_map[module.descriptor] = module.post_process(
                self._hw_data_extractor.get(module.descriptor),
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
        return self.modules.add_node(module, input_handles, output_handle)

    def connect_wrapper(self, wrapper: spiking_modules.HXModuleWrapper):
        """
        Add a wrapper module to the experiment and assign it to the experiments
        modules. In the PyTorch graph the individual module functions assigned
        to the wrapper are then bypassed and only the wrapper's `forward_func`
        is considered when building the PyTorch graph. This functionality is of
        interest if several modules have cyclic dependencies and need to be
        represented by one PyTorch function.

        :param wrapper: The HWModuleWrapper to add to the experiment.
        """
        # Unique modules
        assert len(set(wrapper.modules)) == len(wrapper.modules)

        # Check if modules are already existent
        for other_wrapper in self.modules.wrappers:
            if other_wrapper.contains(wrapper.modules):
                raise ValueError(
                    "You tried to register a wrapper with a group of modules "
                    + "that are partially registered in another group.")

        self.modules.add_wrapper(wrapper)

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

    def _calibrate(self):
        """ """
        for execution_instance in self._execution_instances:
            execution_instance.calibrate()

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
                    not in itertools.chain(
                        self._projections, self._populations):
                module.register_hw_entity()

        # Calibration
        self._calibrate()

        # Generate network graph
        network = self._generate_network_graphs()

        # configure populations
        self._configure_populations()

        # handle runtime
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

        outputs = _hxtorch_spiking.run(
            self._execution_instances.chips, network, inputs,
            self._execution_instances.playback_hooks)

        hw_data = self._get_observables(
            network, outputs, runtime_in_clocks)

        self.modules.reset_changed_since_last_run()

        self._last_run_chip_configs = outputs.pre_execution_chips

        return hw_data, outputs.execution_time_info

    @property
    def last_run_chip_configs(self) -> grenade.signal_flow.OutputData:
        return self._last_run_chip_configs
