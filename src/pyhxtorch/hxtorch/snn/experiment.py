"""
Defining basic types to create hw-executable instances
"""
# pylint: disable=no-member, invalid-name
from typing import Callable, Dict, List, Tuple, Union, Optional
from abc import ABC, abstractmethod
from copy import copy
import itertools
import torch
import numpy as np

from dlens_vx_v3 import hal, halco, sta, lola
import pygrenade_vx as grenade
import _hxtorch
import hxtorch
import hxtorch.snn.modules as snn_module
from hxtorch.snn import handle
from hxtorch.snn.backend.module_manager import BaseModuleManager, ModuleManager

log = hxtorch.logger.get("hxtorch.snn.experiment")


# TODO: Issue: 4007
class NeuronPlacement:
    # TODO: support multi compartment issue #3750
    """
    Tracks assignment of pyNN IDs of HXNeuron based populations to the
    corresponding hardware entities, i.e. LogicalNeuronOnDLS.
    """
    _id_2_ln: Dict[int, halco.LogicalNeuronOnDLS]
    _used_an: List[halco.AtomicNeuronOnDLS]

    def __init__(self):
        self._id_2_ln = {}
        self._used_an = []

    def register_id(
            self,
            neuron_id: Union[List[int], int],
            shape: halco.LogicalNeuronCompartments,
            placement_constraint: Optional[
                Union[List[halco.LogicalNeuronOnDLS],
                      halco.LogicalNeuronOnDLS]] = None):
        """
        Register a new ID to placement
        :param neuron_id: pyNN neuron ID to be registered
        :param shape: The shape of the neurons on hardware.
        :param placement_constraint: A logical neuron or a list of logical
            neurons, each coresponding to one ID in `neuron_id`.
        """
        if not (hasattr(neuron_id, "__iter__")
                and hasattr(neuron_id, "__len__")):
            neuron_id = [neuron_id]
        if placement_constraint is not None:
            self._register_with_logical_neurons(
                neuron_id, placement_constraint)
        else:
            self._register(neuron_id, shape)

    def _register_with_logical_neurons(
            self,
            neuron_id: Union[List[int], int],
            placement_constraint: Union[
                List[halco.LogicalNeuronOnDLS], halco.LogicalNeuronOnDLS]):
        """
        Register neurons with global IDs `neuron_id` with given logical neurons
        `logical_neuron`.
        :param neuron_id: An int or list of ints corresponding to the global
            IDs of the neurons.
        :param placement_constraint: A logical neuron or a list of logical
            neurons, each coresponding to one ID in `neuron_id`.
        """
        if not isinstance(placement_constraint, list):
            placement_constraint = [placement_constraint]
        for idx, ln in zip(neuron_id, placement_constraint):
            if set(self._used_an).intersection(set(ln.get_atomic_neurons())):
                raise ValueError(
                    f"Cannot register LogicalNeuron {ln} since at "
                    + "least one of its compartments are already "
                    + "allocated.")
            self._used_an += ln.get_atomic_neurons()
            assert idx not in self._id_2_ln
            self._id_2_ln[idx] = ln

    def _register(
            self,
            neuron_id: Union[List[int], int],
            shape: halco.LogicalNeuronCompartments):
        """
        Register neurons with global IDs `neuron_id` with creating logical
        neurons implicitly.
        :param neuron_id: An int or list of ints corresponding to the global
            IDs of the neurons.
        :param shape: The shape of the neurons on hardware.
        """
        for idx in neuron_id:
            placed = False
            for anchor in halco.iter_all(halco.AtomicNeuronOnDLS):
                if anchor in self._used_an:
                    continue
                ln = halco.LogicalNeuronOnDLS(shape, anchor)
                fits = True
                for compartment in ln.get_placed_compartments().values():
                    if bool(set(self._used_an) & set(compartment)):
                        fits = False
                if fits:
                    for compartment in ln.get_placed_compartments() \
                            .values():
                        for an in compartment:
                            self._used_an.append(an)
                    self._id_2_ln[idx] = ln
                    placed = True
                    break
            if not placed:
                raise ValueError(f"Cannot register ID {idx}")

    def id2logicalneuron(self, neuron_id: Union[List[int], int]) \
            -> Union[List[halco.LogicalNeuronOnDLS], halco.LogicalNeuronOnDLS]:
        """
        Get hardware coordinate from pyNN int
        :param neuron_id: pyNN neuron int
        """
        try:
            return [self._id_2_ln[idx] for idx in neuron_id]
        except TypeError:
            return self._id_2_ln[neuron_id]


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
            -> Dict[grenade.network.placed_logical.PopulationDescriptor,
                    Tuple[Optional[torch.Tensor], ...]]:
        raise NotImplementedError


class Experiment(BaseExperiment):

    """ Experiment class for describing experiments on hardware """

    def __init__(
            self, mock: bool = False, dt: float = 1e-6,
            hw_routing_func=grenade.network.placed_atomic.build_routing) \
            -> None:
        """
        Instanziate a new experiment, represting an experiment on hardware
        and/or in software.

        :param mock: Indicating whether module is executed on hardware (False)
            or simulated in software (True).
        """
        super().__init__(ModuleManager(), mock=mock, dt=dt)

        # Recording
        self.cadc_recording = {}
        self.has_madc_recording = False

        # Grenade stuff
        self.grenade_hardware_network = None
        self.grenade_hardware_network_graph = None
        self.grenade_logical_network_graph = None
        self.initial_config = None
        self._chip_config = None

        # Configs
        self.injection_pre_static_config = None
        self.injection_pre_realtime = None  # Unused
        self.injection_post_realtime = None  # Unused
        self.injection_inside_realtime_begin = None  # Unused
        self.injection_inside_realtime_end = None  # Unused

        self._populations: List[snn_module.HXModule] = []
        self._projections: List[snn_module.HXModule] = []

        self._batch_size = 0
        self.id_counter = 0

        self.neuron_placement = NeuronPlacement()
        self.hw_routing_func = hw_routing_func

    def clear(self) -> None:
        """
        Reset the experiments's state. Corresponds to creating a new Experiment
        instance.
        """
        self.modules.clear()

        self.cadc_recording = {}
        self.has_madc_recording = False

        self.grenade_hardware_network = None
        self.grenade_hardware_network_graph = None
        self.grenade_logical_network_graph = None
        self.initial_config = None
        self._chip_config = None

        self.injection_pre_static_config = None
        self.injection_pre_realtime = None  # Unused
        self.injection_post_realtime = None  # Unused

        self._populations = []
        self._projections = []

        self._batch_size = 0
        self.id_counter = 0

    def _prepare_static_config(self) -> None:
        """
        Prepares all the static chip config. Accesses the chip object
        initialized by hxtorch.hardware_init and appends corresponding
        configurations to. Additionally this method defines the
        pre_static_config builder injected to grenade at run.
        """
        if self.initial_config is None:
            self._chip_config = _hxtorch.get_chip()
        else:
            self._chip_config = copy(self.initial_config)

        # NOTE: Reserved for inserted config.
        builder = sta.PlaybackProgramBuilder()
        pre_realtime = sta.PlaybackProgramBuilder()
        inside_realtime_begin = sta.PlaybackProgramBuilder()
        inside_realtime_end = sta.PlaybackProgramBuilder()
        post_realtime = sta.PlaybackProgramBuilder()

        self.injection_pre_static_config = builder
        self.injection_pre_realtime = pre_realtime
        self.injection_inside_realtime_begin = inside_realtime_begin
        self.injection_inside_realtime_end = inside_realtime_end
        self.injection_post_realtime = post_realtime
        log.TRACE("Preparation of static config done.")

    def _generate_network_graphs(self) -> Tuple[
            grenade.network.placed_logical.NetworkGraph,
            grenade.network.placed_atomic.NetworkGraph]:
        """
        Generate grenade network graph from the populations and projections in
        modules

        :return: Returns the grenade network graph.
        """
        changed_since_last_run = self.modules.changed_since_last_run()

        log.TRACE(f"Network changed since last run: {changed_since_last_run}")
        if not changed_since_last_run:
            if self.grenade_hardware_network_graph is not None \
                    and self.grenade_logical_network_graph is not None:
                return (self.grenade_logical_network_graph,
                        self.grenade_hardware_network_graph)

        # Create network builder
        network_builder = grenade.network.placed_logical.NetworkBuilder()

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
            cadc_recording = grenade.network.placed_logical.CADCRecording()
            cadc_recording.neurons = list(self.cadc_recording.values())
            network_builder.add(cadc_recording)

        network = network_builder.done()

        network_graph = grenade.network.placed_logical.build_network_graph(
            network)
        self.grenade_logical_network_graph = network_graph
        network = network_graph.hardware_network

        # route network if required
        routing_result = None
        if self.grenade_hardware_network is None \
                or grenade.network.placed_atomic.requires_routing(
                    network, self.grenade_hardware_network):
            routing_result = self.hw_routing_func(network)

        # Keep graph
        self.grenade_hardware_network = network

        # build or update network graph
        if routing_result is not None:
            self.grenade_hardware_network_graph = grenade.network\
                .placed_atomic.build_network_graph(
                    self.grenade_hardware_network, routing_result)
        else:
            grenade.network.placed_atomic.update_network_graph(
                self.grenade_hardware_network_graph,
                self.grenade_hardware_network)

        return (self.grenade_logical_network_graph,
                self.grenade_hardware_network_graph)

    def _configure_populations(self, config: lola.Chip) \
            -> lola.Chip:
        """
        Configure the population on hardware.

        :param config: The config object to write the population confiuration
            at.

        :return: Returns the config object with the configuration appended.
        """
        # Make sure experiment holds chip config
        assert config is not None

        pop_changed_since_last_run = any(
            m.changed_since_last_run for m in self._populations)
        if not pop_changed_since_last_run:
            return config

        for module in self._populations:
            if not isinstance(module, snn_module.Neuron):
                continue
            log.TRACE(f"Configure population '{module}'.")
            for in_pop_id, unit_id in enumerate(module.unit_ids):
                coord = self.neuron_placement.id2logicalneuron(unit_id)
                config.neuron_block = module.configure_hw_entity(
                    in_pop_id, config.neuron_block, coord)
                log.TRACE(
                    f"Configured neuron at coord {coord}.")

        return config

    def _generate_inputs(
        self, network_graph: grenade.network.placed_logical.NetworkGraph,
        hardware_network_graph: grenade.network.placed_atomic.NetworkGraph) \
            -> grenade.signal_flow.IODataMap:
        """
        Generate external input events from the routed network graph
        representation.
        """
        assert hardware_network_graph.event_input_vertex is not None
        if hardware_network_graph.event_input_vertex is None:
            return grenade.signal_flow.IODataMap()

        # Make sure all batch sizes are equal
        sizes = [
            handle.observable_state.shape[1] for handle in
            self.modules.input_data()]
        assert all(sizes)
        self._batch_size = sizes[0]

        input_generator = grenade.network.placed_logical.InputGenerator(
            network_graph, hardware_network_graph, self._batch_size)
        for module in self._populations:
            in_handle = [
                e["handle"] for _, _, e in self.modules.graph.in_edges(
                    self.modules.get_id_by_module(module), data=True)].pop()
            module.add_to_input_generator(in_handle, input_generator)

        return input_generator.done()

    def _generate_playback_hooks(self) \
            -> grenade.signal_flow.ExecutionInstancePlaybackHooks:
        """ Handle injected config (not suppored yet) """
        assert self.injection_pre_static_config is not None
        assert self.injection_pre_realtime is not None
        assert self.injection_inside_realtime_begin is not None
        assert self.injection_inside_realtime_end is not None
        assert self.injection_post_realtime is not None
        pre_static_config = sta.PlaybackProgramBuilder()
        pre_realtime = sta.PlaybackProgramBuilder()
        inside_realtime_begin = sta.PlaybackProgramBuilder()
        inside_realtime_end = sta.PlaybackProgramBuilder()
        post_realtime = sta.PlaybackProgramBuilder()
        pre_static_config.copy_back(self.injection_pre_static_config)
        pre_realtime.copy_back(self.injection_pre_realtime)
        inside_realtime_begin.copy_back(self.injection_inside_realtime_begin)
        inside_realtime_end.copy_back(self.injection_inside_realtime_end)
        post_realtime.copy_back(self.injection_post_realtime)
        return grenade.signal_flow.ExecutionInstancePlaybackHooks(
            pre_static_config, pre_realtime, inside_realtime_begin,
            inside_realtime_end, post_realtime)

    def _get_population_observables(
            self, network_graph: grenade.network.placed_logical.NetworkGraph,
            hardware_network_graph: grenade.network.placed_atomic.NetworkGraph,
            result_map: grenade.signal_flow.IODataMap, runtime) -> Dict[
                grenade.network.placed_logical.PopulationDescriptor,
                np.ndarray]:
        """
        Takes the greade network graph and the result map returned by grenade
        after experiment execution and returns a data map where for each
        population descriptor of registered populations the population-specific
        hardware observables are represented as Optional[torch.Tensor]s.
        Note: This function calles the modules `post_process` method.
        :param network_graph: The logical grenade network graph describing the
            logic of th experiment.
        :param hardware_network_graph: The grenade hardware network graph
            describing the experiment on hardware.
        :param result_map: The result map returned by grenade holding all
            recorded hardware observables.
        :param runtime: The runtime of the experiment executed on hardware in
            ms.
        :returns: Returns the data map as dict, where the keys are the
            population descriptors and values are tuples of values returned by
            the correpsonding module's `post_process` method.
        """
        # Get hw data
        hw_spike_times = hxtorch.snn.extract_spikes(
            result_map, network_graph, hardware_network_graph, runtime)
        hw_cadc_samples = hxtorch.snn.extract_cadc(
            result_map, network_graph, hardware_network_graph, runtime)
        hw_madc_samples = hxtorch.snn.extract_madc(
            result_map, network_graph, hardware_network_graph, runtime)

        # Data maps
        data_map: Dict[
            grenade.network.placed_logical.PopulationsDescriptor,
            Tuple[torch.Tensor]] = {}  # pylint: disable=c-extension-no-member

        # Map populations to data
        for module in self._populations:
            if isinstance(module, snn_module.InputNeuron):
                continue
            data_map[module.descriptor] = module.post_process(
                hw_spike_times.get(module.descriptor),
                hw_cadc_samples.get(module.descriptor),
                hw_madc_samples.get(module.descriptor))

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

    def wrap_modules(self, modules: List[snn_module.HXModule],
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
            snn_module.HXModuleWrapper(self, modules, func))

    def register_population(self, module: snn_module.HXModule) -> None:
        """
        Register a module as population.

        :param module: The module to register as population.
        """
        self._populations.append(module)

    def register_projection(self, module: snn_module.HXModule) -> None:
        """
        Register a module as projection.

        :param module: The module to register as projection.
        """
        self._projections.append(module)

    def get_hw_results(self, runtime: Optional[int]) \
            -> Dict[grenade.network.placed_logical.PopulationDescriptor,
                    Tuple[Optional[torch.Tensor], ...]]:
        """
        Executes the experiment in mock or on hardware using the information
        added to the experiment for a time given by `runtime` and returns a
        dict of hardware data represented as PyTorch data types.

        :param runtime: The runtime of the experiment on hardware in ms.

        :returns: Returns the data map as dict, where the keys are the
            population descriptors and values are tuples of values returned by
            the correpsonding module's `post_process` method.
        """
        if not self.mock and self._chip_config is None:
            self._prepare_static_config()

        # Preprocess layer
        self.modules.pre_process(self)

        # In mock-mode nothing to do here
        if self.mock:
            return {}

        # Register HW entity
        for module in self.modules.nodes:
            if hasattr(module, "register_hw_entity") and module \
                    not in itertools.chain(self._projections,
                                           self._populations):
                module.register_hw_entity()

        # Generate network graph
        logical_network, hardware_network = self._generate_network_graphs()

        # Make sure chip config is present (by calling prepare_static_chfig)
        assert self._chip_config is not None

        # configure populations
        self._chip_config = self._configure_populations(self._chip_config)

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
        inputs = self._generate_inputs(logical_network, hardware_network)
        inputs.runtime = [{
            grenade.signal_flow.ExecutionInstance(): runtime_in_clocks
        }] * self._batch_size
        log.TRACE(f"Registered runtimes: {inputs.runtime}")

        outputs = hxtorch.snn.grenade_run(
            self._chip_config, hardware_network, inputs,
            self._generate_playback_hooks())

        hw_data = self._get_population_observables(
            logical_network, hardware_network, outputs, runtime_in_clocks)

        self.modules.reset_changed_since_last_run()

        return hw_data
