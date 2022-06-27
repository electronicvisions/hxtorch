"""
Defining basic types to create hw-executable instances
"""
# pylint: disable=no-member, invalid-name
from typing import Dict, Final, List, Tuple, Union, Optional
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
import hxtorch.snn.handle as handle
from hxtorch.snn.backend.nodes import Node
from hxtorch.snn.backend.module_manager import BaseModuleManager, ModuleManager

log = hxtorch.logger.get("hxtorch.snn.instance")


# TODO: Issue: 4007
class NeuronPlacement:

    """ Define neuron placement on hardware """

    _id_2_an: Dict[int, halco.AtomicNeuronOnDLS]
    _permutation: List[halco.AtomicNeuronOnDLS]
    _max_num_entries: Final[int] = halco.AtomicNeuronOnDLS.size
    default_permutation: Final[List[int]] = range(halco.AtomicNeuronOnDLS.size)

    def __init__(self, permutation: List[int] = None) -> None:
        if permutation is None:
            permutation = self.default_permutation
        self._id_2_an = dict()
        self._permutation = self._check_and_transform(permutation)

    def register_id(self, neuron_id: int) -> None:
        """
        Register a new ID to placement

        :param neuron_id: neuron ID to be registered
        """
        if not (hasattr(neuron_id, "__iter__")
                and hasattr(neuron_id, "__len__")):
            neuron_id = [neuron_id]
        if len(self._id_2_an) + len(neuron_id) > len(self._permutation):
            raise ValueError(
                f"Cannot register more than {len(self._permutation)} IDs")
        for idx in neuron_id:
            self._id_2_an[idx] = self._permutation[len(self._id_2_an)]

    def id2atomicneuron(self, neuron_id: Union[List[int], int]) \
            -> Union[List[halco.AtomicNeuronOnDLS], halco.AtomicNeuronOnDLS]:
        """
        Get hardware coordinate from ID

        :param neuron_id: neuron ID
        """
        try:
            return [self._id_2_an[idx] for idx in neuron_id]
        except TypeError:
            return self._id_2_an[neuron_id]

    def id2hwenum(self, neuron_id: Union[List[int], int]) \
            -> Union[List[int], int]:
        """
        Get hardware coordinate as plain int from atomic neuron.

        :param neuron_id: neuron ID
        """
        atomic_neuron = self.id2atomicneuron(neuron_id)
        try:
            return [int(idx.toEnum()) for idx in atomic_neuron]
        except TypeError:
            return int(atomic_neuron.toEnum())

    @staticmethod
    def _check_and_transform(lut: list) -> list:
        """
        """
        cell_id_size = NeuronPlacement._max_num_entries
        if len(lut) > cell_id_size:
            raise ValueError("Too many elements in HW LUT.")
        if len(lut) > len(set(lut)):
            raise ValueError("Non unique entries in HW LUT.")
        permutation = []
        for neuron_idx in lut:
            if not 0 <= neuron_idx < cell_id_size:
                raise ValueError(
                    f"NeuronPermutation list entry {neuron_idx} out of range. "
                    + f"Needs to be in range [0, {cell_id_size - 1}]")
            coord = halco.AtomicNeuronOnDLS(halco.common.Enum(neuron_idx))
            permutation.append(coord)

        return permutation


class BaseInstance(ABC):

    def __init__(self, modules: BaseModuleManager, mock: bool, dt: float) \
            -> None:
        self.mock = mock
        self.modules = modules
        self.dt = dt

    @abstractmethod
    def connect(self, module: torch.nn.Module,
                input_handle: handle.TensorHandle,
                output_handle: handle.TensorHandle) -> Node:
        raise NotImplementedError

    @abstractmethod
    def get_hw_results(self, runtime: Optional[int]) \
            -> Dict[grenade.PopulationDescriptor,
                    Tuple[Optional[torch.Tensor], ...]]:
        raise NotImplementedError


class Instance(BaseInstance):

    """ Instance class for describing experiments on hardware """

    def __init__(self, mock: bool = False, dt: float = 1e-6) -> None:
        """
        Instanziate a new instance, represting an experiment on hardware and/or
        in software.

        :param mock: Indicating whether module is executed on hardware (False)
            or simulated in software (True).
        """
        super().__init__(ModuleManager(), mock=mock, dt=dt)

        # Recording
        self.cadc_recording = dict()
        self.has_madc_recording = False

        # Grenade stuff
        self.grenade_network = None
        self.grenade_network_graph = None
        self.initial_config = None
        self._chip_config = None

        # Configs
        self.injection_pre_static_config = None
        self.injection_pre_realtime = None  # Unused
        self.injection_post_realtime = None  # Unused
        self.injection_inside_realtime_begin = None  # Unused
        self.injection_inside_realtime_end = None  # Unused

        self._batch_size = 0
        self.id_counter = 0

        self.neuron_placement = NeuronPlacement()

    def clear(self) -> None:
        """
        Reset the instance's state. Corresponds to creating a new Instance
        instance.
        """
        self.modules.clear()

        self.cadc_recording = dict()
        self.has_madc_recording = False

        self.grenade_network = None
        self.grenade_network_graph = None
        self.initial_config = None
        self._chip_config = None

        self.injection_pre_static_config = None
        self.injection_pre_realtime = None  # Unused
        self.injection_post_realtime = None  # Unused

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

    def _generate_network_graph(self) -> grenade.NetworkGraph:
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
        network_builder = grenade.NetworkBuilder()

        # Add populations
        for node in self.modules.populations:
            node.descriptor = node.module.add_to_network_graph(network_builder)
        # Add projections
        for node in self.modules.projections:
            pre_pop = self.modules.pre_populations(node)
            post_pop = self.modules.post_populations(node)
            assert len(pre_pop) == 1, "On hardware, a projection can only " \
                "have one source population."
            assert len(post_pop) == 1, "On hardware, a projection can only " \
                "have one target population."
            node.descriptor = node.module.add_to_network_graph(
                pre_pop.pop().descriptor, post_pop.pop().descriptor,
                network_builder)

        # Add CADC recording
        if self.cadc_recording:
            cadc_recording = grenade.CADCRecording()
            cadc_recording.neurons = list(self.cadc_recording.values())
            network_builder.add(cadc_recording)

        network = network_builder.done()

        # route network if required
        routing_result = None
        if self.grenade_network is None \
                or grenade.requires_routing(network, self.grenade_network):
            routing_result = grenade.build_routing(network)

        # Keep graph
        self.grenade_network = network

        # build or update network graph
        if routing_result is not None:
            self.grenade_network_graph = grenade.build_network_graph(
                self.grenade_network, routing_result)
        else:
            grenade.update_network_graph(
                self.grenade_network_graph, self.grenade_network)

        return self.grenade_network_graph

    def _configure_populations(self, config: lola.Chip) \
            -> lola.Chip:
        """
        Configure the population on hardware.

        :param config: The config object to write the population confiuration
            at.

        :return: Returns the config object with the configuration appended.
        """
        # Make sure instance holds chip config
        assert config is not None

        pop_changed_since_last_run = any(
            node.module.changed_since_last_run for node in self.modules)
        if not pop_changed_since_last_run:
            return config

        for node in self.modules.populations:
            if not isinstance(node.module, snn_module.Neuron):
                continue
            log.TRACE(f"Configure population '{node.module}'.")
            for in_pop_id, unit_id in enumerate(node.module.unit_ids):
                coord = self.neuron_placement.id2atomicneuron(unit_id)
                atomic_neuron = node.module.configure_hw_entity(
                    in_pop_id, config.neuron_block.atomic_neurons[coord])
                config.neuron_block.atomic_neurons[coord] = atomic_neuron
                log.TRACE(
                    f"Configured neuron at coord {coord}:\n{atomic_neuron}.")

        return config

    def _generate_inputs(self, network_graph: grenade.NetworkGraph) \
            -> grenade.IODataMap:
        """
        Generate external input events from the routed network graph
        representation.
        """
        assert network_graph.event_input_vertex is not None
        if network_graph.event_input_vertex is None:
            return grenade.IODataMap()

        # Make sure all batch sizes are equal
        sizes = [
            node.input_handle[0].observable_state.shape[0] for node in
            self.modules.inputs()]
        assert all(sizes)
        self._batch_size = sizes[0]

        input_generator = grenade.InputGenerator(
            network_graph, self._batch_size)
        for node in self.modules.populations:
            node.module.add_to_input_generator(
                node.input_handle, input_generator)

        return input_generator.done()

    def _generate_playback_hooks(self) \
            -> grenade.ExecutionInstancePlaybackHooks:
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
        return grenade.ExecutionInstancePlaybackHooks(
            pre_static_config, pre_realtime, inside_realtime_begin,
            inside_realtime_end, post_realtime)

    def _get_population_observables(self, network_graph: grenade.NetworkGraph,
                                    result_map: grenade.IODataMap,
                                    runtime) \
            -> Dict[grenade.PopulationDescriptor, np.ndarray]:
        """
        Takes the greade network graph and the result map returned by grenade
        after experiment execution and returns a data map where for each
        population descriptor of registered populations the population-specific
        hardware observables are represented as Optional[torch.Tensor]s.
        Note: This function calles the modules `post_process` method.

        :param network_graph: The grenade network graph describing the
            experiment on hardware.
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
            result_map, network_graph, runtime)
        hw_cadc_samples = hxtorch.snn.extract_cadc(
            result_map, network_graph, runtime)
        hw_madc_samples = hxtorch.snn.extract_madc(
            result_map, network_graph, runtime)

        # Data maps
        data_map: Dict[
            grenade.PopulationsDescriptor, Tuple[torch.Tensor]] = dict()  # pylint: disable=c-extension-no-member

        # Map populations to data
        for pop in self.modules.populations:
            if isinstance(pop.module, snn_module.InputNeuron):
                continue
            data_map[pop.descriptor] = pop.module.post_process(
                hw_spike_times.get(pop.descriptor),
                hw_cadc_samples.get(pop.descriptor),
                hw_madc_samples.get(pop.descriptor))

        return data_map

    def connect(self, module: torch.nn.Module,
                input_handle: handle.TensorHandle,
                output_handle: handle.TensorHandle) -> Node:
        """
        Add an module to the instance and connect it to other instance
        modules via input and output handles.

        :param module: The HXModule to add to the instance.
        :param input_handle: The TensorHandle serving as input to the module
            (its obsv_state).
        :param output_handle: The TensorHandle outputted by the module,
            serving as input to subsequent HXModules.
        """
        return self.modules.add(module, input_handle, output_handle)

    def register_population(self, module: snn_module.HXModule) -> None:
        """
        Register a module as population.

        :param module: The module to register as population.
        """
        self.modules.populations.add(self.modules.get_node(module))

    def register_projection(self, module: snn_module.HXModule) -> None:
        """
        Register a module as projection.

        :param module: The module to register as projection.
        """
        self.modules.projections.add(self.modules.get_node(module))

    def get_hw_results(self, runtime: Optional[int]) \
            -> Dict[grenade.PopulationDescriptor,
                    Tuple[Optional[torch.Tensor], ...]]:
        """
        Executes the experiment in mock or on hardware using the information
        added to the instance for a time given by `runtime` and returns a dict
        of hardware data represented as PyTorch data types.

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
            return dict()

        # Register HW entity
        for node in self.modules:
            if hasattr(node.module, "register_hw_entity") and node \
                    not in itertools.chain(self.modules.projections,
                                           self.modules.populations):
                node.module.register_hw_entity()

        # Generate network graph
        network_graph = self._generate_network_graph()

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
        inputs = self._generate_inputs(network_graph)
        inputs.runtime = {
            grenade.ExecutionInstance(): self._batch_size * [runtime_in_clocks]
        }
        log.TRACE(f"Registered runtimes: {inputs.runtime}")

        outputs = hxtorch.snn.grenade_run(
            self._chip_config, network_graph, inputs,
            self._generate_playback_hooks())

        hw_data = self._get_population_observables(
            network_graph, outputs, runtime_in_clocks)

        self.modules.reset_changed_since_last_run()

        return hw_data
