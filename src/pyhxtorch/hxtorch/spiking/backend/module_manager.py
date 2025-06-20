"""
Define graph-based data structure managing Nodes
"""
from typing import Any, Dict, Tuple, List, Union, Set, Optional
from abc import ABC, abstractmethod
import networkx as nx

import hxtorch.spiking.modules as spiking_module

Source = Tuple[Any]
Target = Any
Module = Union[spiking_module.HXBaseExperimentModule, Any]
Wrapper = Union[spiking_module.HXModuleWrapper, Any]


class BaseModuleManager(ABC):
    """ Abstract base class for module manager """

    def __init__(self):
        """ """
        self.graph: nx.DiGraph = nx.DiGraph()
        self.prev_graph: nx.DiGraph = nx.DiGraph()
        self.nodes: Dict[Module, int] = {}
        self.wrappers: Dict[Wrapper, int] = {}

    def __str__(self) -> str:
        """ Add proper object string """
        string = "Modules (node_id, module):\n"
        if not self.nodes:
            string += "\tNone\n"
        else:
            for module, node in self.nodes.items():
                string += f"\t{node}: {module}\n"
        string += "Wrappers (wrapper_id, wrapper):\n"
        if not self.wrappers:
            string += "\tNone\n"
        else:
            for wrapper, node in self.wrappers.items():
                string += f"\t{node}: {wrapper}\n"
        string += "Connections:\n"
        for node in self.graph.nodes():
            string += f"\tNode {node}:\n"
            string += "\t\tSources: " \
                + f"{list(self.graph.predecessors(node))}\n"
            string += "\t\tTargets: " \
                + f"{list(self.graph.successors(node))}\n"
        return string

    def clear(self):
        """
        Clear the internal graph and remove the module -> node and
        wrapper -> node mapping
        """
        self.prev_graph = self.graph.copy()
        self.graph = nx.DiGraph()

    @abstractmethod
    def add_node(self, module: Module, sources: Source, target: Target):
        """
        Adds a new module to the manager. This method adds a node to the
        internal graph to represent the module. It assigns edges to this node
        holding the data in `sources`, resp. `target`.
        :param module: Module to represented in to the graph.
        :param sources: The sources to the node representing `module`.
        :param targets: The targets of the node representing `module`.
        """
        raise NotImplementedError

    @abstractmethod
    def add_wrapper(self, wrapper: Wrapper):
        """
        Adds a new wrapper to the manager. This must be called after all
        modules wrapped by this wrapper are represented in the graph.
        internal graph to represent the module. It assigned edges to this node
        holding the data in `sources`, resp. `target`.
        :param module: Module to represented in to the graph.
        """
        raise NotImplementedError

    @abstractmethod
    def done(self) -> List[Tuple[Module, Source, Target]]:
        """
        Create a list of elements of form (module, sources, targets), which can
        be looped in the correct order.
        :return: Returns the a list of nodes to be executed.
        """
        raise NotImplementedError


# We allow u, v as variable names to be consistent with networkx
# pylint: disable=invalid-name
class ModuleManager(BaseModuleManager):
    """ Object representing all nodes in a graph-like data structure """

    # Types that are recognized as populations on hardware
    _default_input_type = spiking_module.InputNeuron
    # Types that are recognized as populations on hardware
    _population_types = (
        spiking_module.Population, spiking_module.InputPopulation)

    def __init__(self):
        """
        Initialize a `Modules` object. This object holds a list of `nodes`.
        """
        super().__init__()
        self._inputs: Dict = {}
        self._open_sources: Set = set()
        self._open_targets: Set = set()
        self._graph_hash: Optional[str] = None

    def __str__(self):
        """ Append string """
        string = super().__str__()
        string += "Inputs (Module, Input Module):\n"
        if not self._inputs:
            string += "\tNone\n"
        else:
            for in_module, target_module in self._inputs.items():
                string += f"\t{in_module}: {target_module}\n"
        string += f"Open sources:\n\t{list(self._open_sources)}\n"
        string += f"Open targets:\n\t{list(self._open_targets)}"
        return string

    def clear(self):
        """
        Override clear to also clear open sources and open targets. This method
        resets the Manager without removing implicitly created input modules
        such that they can be reused.
        """
        super().clear()
        self._open_sources: Set = set()
        self._open_targets: Set = set()

    def get_node_id(self) -> int:
        """
        Get the ID of the next node to add.
        :returns: Returns the next usable ID of a node to add.
        """
        return len(self.nodes)

    def get_wrapper_id(self) -> int:
        """
        Get the ID of the next wrapper to add.
        :returns: Returns the next usable ID of a wrapper to add.
        """
        return f"w_{len(self.wrappers)}"

    def get_module_by_id(self, node_id: int):
        """
        Finds the module of the node with ID `node_id` and returns it. If
        multiple modules assigned to this ID are found, only the first one is
        returned. However, this should never be the case and if so, it is a
        bug.
        :param node_id: The ID of the node to find the module for.
        :returns: Returns the corresponding module.
        """
        return [key for key, val in self.nodes.items() if val == node_id].pop()

    def get_wrapper_by_id(self, wrapper_id: int):
        """
        Finds the wrapper of the node with ID `wrapper_id` and returns it. If
        multiple wrapper assigned to this ID are found, only the first one is
        returned. However, this should never be the case and if so, it is a
        bug.
        :param wrapper_id: The ID of the node to find the module for.
        :returns: Returns the corresponding wrapper.
        """
        return [
            key for key, val in self.wrappers.items()
            if val == wrapper_id].pop()

    def get_id_by_module(self, module: Module) -> Optional[int]:
        """
        Finds the ID of the node which corresponds to module `module`. If no ID
        is found, `None` is returned.
        :param module: The module to find the node ID for.
        :returns: Returns the node ID or `None` if no ID is found.
        """
        return self.nodes.get(module)

    def get_id_by_wrapper(self, wrapper: Wrapper) -> Optional[int]:
        """
        Finds the ID of the wrapper which corresponds to wrapper `wrapper`. If
        no ID is found, `None` is returned.
        :param wrapper: The wrapper module to find the node ID for.
        :returns: Returns the node ID or `None` if no ID is found.
        """
        return self.wrappers.get(wrapper)

    def changed_since_last_run(self) -> bool:
        """
        Check if any module is marked dirty.
        :return: Returns true if at least one module has been marked dirty.
        """
        topology_changed = self._graph_hash != nx.weisfeiler_lehman_graph_hash(
            self.graph)
        modules_changed = any(
            module.changed_since_last_run for module in self.nodes)
        return topology_changed or modules_changed

    def reset_changed_since_last_run(self):
        """
        Restes all registered modules changed_since_last_run flags.
        """
        self._graph_hash = nx.weisfeiler_lehman_graph_hash(self.graph)
        for module in self.nodes:
            module.reset_changed_since_last_run()

    def has_module(self, module: Module) -> bool:
        """
        Checks whether the module `module` is already registered within the
        graph.
        :param module: The module to check its existence for.
        :return: Returns a bool indicating whether the module exists or not.
        """
        return module in self.nodes

    def find_edges(self, handle: Any) -> List[int]:
        """
        Find all edges with data associated with `handle`.
        :param handle: The edge data to match against.
        :return: Returns a list of all edges in the graph which hold the same
            data `handle`.
        """
        return [
            (u, v) for u, v, e in self.graph.edges(data=True)
            if id(e['handle']) == id(handle)]

    def input_data(self) -> List[Any]:
        """
        Finds all edge data associated to edges with a source node in
        _open_sources, those nodes are roots.
        :return: Returns a list of all input data.
        """
        return [
            e['handle'] for u, v, e in self.graph.edges(data=True)
            if u in self._open_sources]

    def add_node(self, module: Module, sources: Source, target: Target):
        """
        Adds a new module to the manager. This method adds a node to the
        internal graph to represent the module. It assigned edges to this node
        holding the data in `sources`, resp. `target`.
        :param module: Module to represented in to the graph.
        :param sources: The sources to the node representing `module`.
        :param targets: The targets of the node representing `module`.
        """
        node_id = self.get_id_by_module(module)
        if node_id is None:
            node_id = self.get_node_id()
            self.nodes.update({module: node_id})

        # Cast to tuple
        if not isinstance(sources, tuple):
            sources = (sources,)

        # Handle sources
        for i, source in enumerate(sources):
            edges = self.find_edges(source)
            for u, v in edges:
                if v in self._open_targets:
                    self._open_targets.remove(v)
                    self.graph.remove_node(v)
                    break
            if not edges:
                u = f"s_{node_id}_{i}"
                self._open_sources.add(u)
            self.graph.add_edge(u, node_id, handle=source)

        # Handle targets
        # TODO: Possibly allow for multiple targets
        edge = self.find_edges(target)
        if edge:
            u, v = edge[0]
            if u in self._open_sources and not u == node_id:
                self._open_sources.remove(u)
                self.graph.remove_node(u)
        if not edge:
            v = f"t_{node_id}"
            self._open_targets.add(v)
        self.graph.add_edge(node_id, v, handle=target)

    def add_wrapper(self, wrapper: Wrapper):
        """
        Adds a new wrapper to the manager. This must be called after all
        modules wrapped by this wrapper are represented in the graph.
        internal graph to represent the module. It assigned edges to this node
        holding the data in `sources`, resp. `target`.
        :param module: Module to represented in to the graph.
        """
        wrapper_id = self.get_id_by_wrapper(wrapper)
        # Get existing node or create new node
        if wrapper_id is None:
            wrapper_id = self.get_wrapper_id()
        self.wrappers.update({wrapper: wrapper_id})

    def _get_populations(self, module: Module, target: bool = False) \
            -> List[Module]:
        """
        Find the target, resp. source populations of module `module`, i.e.
        modules which are of type self._population_types.
        :param module: The module to find the source population (target=False)
            or the target populations for(target=True)
        :param target: A bool indicating whether the search is performed for
            source or target populations.
        :return: Returns a list of source, resp. target populations of module.
        """
        method = self.graph.successors if target \
            else self.graph.predecessors
        id_module = dict(zip(self.nodes.values(), self.nodes.keys()))
        stack = list(method(self.get_id_by_module(module)))

        pops = []
        while True:
            if not stack:
                return pops
            node_id = stack.pop()
            module = id_module[node_id]
            if isinstance(module, self._population_types):
                pops.append(module)
                continue
            stack += list(method(node_id))
        return pops

    def source_populations(self, module: Module):
        """
        Find the source populations of module `module`, i.e. modules which are
        of type self._population_types.
        :param module: The module to find the source population.
        :return: Returns a list of source populations of module.
        """
        return self._get_populations(module, False)

    def target_populations(self, module: Module):
        """
        Find the target populations of module `module`, i.e. modules which are
        of type self._population_types.
        :param module: The module to find the target populations for.
        :return: Returns a list of target populations of module.
        """
        return self._get_populations(module, True)

    def _handle_inputs(self, instance):
        """
        On hardware, synapse have external populations as sources. Hence we
        have to augment Synapses with InputNeurons.
        """
        # Replace open sources with input modules
        for u in self._open_sources:
            vs = list(self.graph.successors(u))
            assert len(vs) == 1
            v = vs.pop()

            module = self.get_module_by_id(v)
            if not isinstance(module, spiking_module.Projection):
                continue

            in_module = self._inputs.get(module)
            if not in_module:
                in_module = self._default_input_type(
                    module.in_features, instance, module.execution_instance)
                self._inputs.update({module: in_module})
                self.nodes.update({in_module: self.get_node_id()})

            # Update graph, i.e. forward input module handles
            source = self.graph.get_edge_data(u, v)["handle"]
            target = in_module.output_type()
            in_id = self.nodes[in_module]
            self.graph.add_edge(u, in_id, handle=source)
            self.graph.add_edge(in_id, v, handle=target)
            self.graph.remove_edge(u, v)

    def _handle_dropout_mask(self):
        """
        Method to back-assign spiking mask from Dropout layers to the previous
        Neuron layer in order to configure the neurons appropriately on
        hardware.

        Note: BatchDropout layers have to be preceded by a Neuron layer.
        """
        for module, u in self.nodes.items():
            if u in self._open_sources:
                continue
            if isinstance(module, spiking_module.BatchDropout):
                pre_nodes = list(self.graph.predecessors(u))
                if len(pre_nodes) != 1:
                    raise TypeError(
                        "The BatchDropout module is only allowed to "
                        "preceed one module.")
                pre_module = self.get_module_by_id(pre_nodes.pop())
                if not isinstance(pre_module, spiking_module.AELIF):
                    raise TypeError(
                        "The BatchDropout module is only allowed to "
                        "succeed a Neuron module.")
                pre_module.mask = module.set_mask()

    def _handle_wrappers(self):
        """
        Handle registered wrappers. This method needs to be called after all
        modules are registered. It replaces all nodes associated with the nodes
        in a given wrapper in the graph. All internal target handles are
        gathered in the wrapper's node attributes 'targets'. All external input
        sources to the wrapper are gathered in the node attributes 'sources'.
        """
        # Keep original graph
        for wrapper, w_id in self.wrappers.items():
            self.graph.add_node(w_id, is_wrapper=True)
            ids = [self.nodes[m] for m in wrapper.modules.values()]

            targets, sources = [], []
            for n_id in ids:
                for u, _, e in self.graph.in_edges(n_id, data=True):
                    if u not in ids:
                        sources.append(e["handle"])
                        self.graph.add_edge(u, w_id, **e)
                for _, v, e in self.graph.out_edges(n_id, data=True):
                    if v not in ids:
                        self.graph.add_edge(w_id, v, **e)
                    if (id(e["handle"]) not in
                            [id(target) for target in targets]):
                        targets.append(e["handle"])
            self.graph.remove_nodes_from(ids)
            self.graph.nodes[w_id]["sources"] = sources
            self.graph.nodes[w_id]["targets"] = targets

    def _order(self) -> List[Tuple[Module, Source, Tuple[Target]]]:
        """
        This method checks whether the internal graph representation has no
        cyclic dependencies. Cyclic dependencies need to be wrapped manually at
        the moment.
        TODO: Introduce implicit module wrapping here.
        Further, a topological sort is performed in order to return the nodes
        in an order in which all modules can be executed successively.
        :return: Returns a list of tuples of shape (module, sources, targets),
            where sources is a tuple of all input sources and targets is a
            tuple of all output targets of module `module`.
        """
        # Check for cycles in graph
        graph = self.graph.to_undirected(as_view=True)
        if nx.cycle_basis(graph):
            raise ValueError(
                "Encountered cyclic dependencies in the network. At the"
                " moment, cyclic dependencies are expected to be wrapped"
                " manually.")
        # Order nodes
        nodes = []
        for n in nx.topological_sort(self.graph):
            if n in self._open_sources or n in self._open_targets:
                continue
            if "is_wrapper" in self.graph.nodes[n]:
                source = self.graph.nodes[n]["sources"]
                target = tuple(self.graph.nodes[n]["targets"])
                module = self.get_wrapper_by_id(n)
            else:
                source, target = [], []
                for _, _, e in self.graph.in_edges(n, data=True):
                    source.append(e["handle"])
                for _, _, e in self.graph.out_edges(n, data=True):
                    handle = e["handle"]
                    if id(handle) not in [id(trgt) for trgt in target]:
                        target.append(handle)
                assert len(target) == 1
                target = target.pop()
                module = self.get_module_by_id(n)
            nodes.append((module, tuple(source), target))
        return nodes

    def pre_process(self, instance):
        """
        Handle all preprocessing needed prior to hardware execution.
        This includes input module injection as well as setting the dropout
        masks.
        """
        self._handle_dropout_mask()
        if instance.mock:
            return
        self._handle_inputs(instance)

    def done(self):
        """
        Create a list of elements of form (module, sources, targets), which can
        be looped in the correct order.
        :param instance: The instance to work on.
        :return: Returns the ordered modules in the form (module, sources,
            targets).
        """
        self._handle_wrappers()
        nodes = self._order()
        self.clear()
        return nodes
