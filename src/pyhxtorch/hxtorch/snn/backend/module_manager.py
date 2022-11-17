"""
Define graph-based datastructure managing Nodes
"""
from typing import Dict, Tuple, List, Union, Set
from abc import ABC, abstractmethod

import hxtorch.snn.modules as snn_module
from hxtorch.snn import handle
from hxtorch.snn.backend.nodes import Node


class BaseModuleManager(ABC):
    """ Abstract base class for module manager """

    # Types that are recognized as populations on hardware
    _population_types: Tuple[snn_module.HXModule]
    # Types that are recognized as projections on hardware
    _projection_types: Tuple[snn_module.HXModule]

    def __init__(self) -> None:
        """ """
        self._nodes: List[Node] = []

    def __str__(self):
        """ Add proper object string """
        loose_string = ""
        connected_string = ""
        for node in self._nodes:
            if node.pre or node.post:
                connected_string += f"\n{node}"
            else:
                loose_string += f"\n{node}"
        loose_string = " None" if loose_string == "" else loose_string
        connected_string = " None" if connected_string == "" \
            else connected_string
        string = f"Connected Nodes:{connected_string}" \
            + f"\nLoose Nodes:{loose_string}"
        return string

    def __getitem__(self, index: int) -> Node:
        """
        Retrieve a node in the `Modules` graph at index `index`.

        :param index: The index at which to access the node.
        """
        return self._nodes[index]

    def __len__(self):
        """
        The length of the manager corresponds to the length of the nodes
        registered.

        :return: Returns the number of nodes registered.
        """
        return len(self._nodes)

    @abstractmethod
    def add(self, module: snn_module.HXModule,
            input_handles: Union[
                Tuple[handle.TensorHandle], handle.TensorHandle],
            output_handle: handle.TensorHandle) -> Node:
        """
        Adds a new module to the manager without creating edges. This method
        created a node implictly (or update an existing node), keeps it and
        finally returns it.

        :param module: HXModule to add to the graph.
        :param input_handles: The input HXHandle (or a tuple of handles),
            passed to the modules forward method.
        :param output_handle: The output HXHandle returned by the modules
            forward method.

        :return: Returns the created/updates node.
        """
        raise NotImplementedError

    @abstractmethod
    def pre_process(self, instance) -> None:
        """
        A method evoked by Instance, request the module manager to prepare the
        modules.
        """
        raise NotImplementedError

    @abstractmethod
    def ordered(self) -> List[Node]:
        """
        Create a list of nodes which can be looped in the correct order. This
        means for each node in the list its pre-nodes are at previous postions.

        :return: Returns the ordered list.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """ Delete all nodes from the graph. """
        self._nodes = []

    def leafs(self) -> List[Node]:
        """
        Get the list of leaf nodes. Those are defined as the nodes in the graph
        which are not connected to post-nodes.

        :return: Returns the list of the graphs leafs.
        """
        return [n for n in self._nodes if not n.post]

    def inputs(self) -> List[Node]:
        """
        Get the list of input nodes. Those are defined as the nodes in the
        graph which do not have any pre-nodes assigned.

        :return: Returnes the list of input nodes of the graph.
        """
        return [node for node in self._nodes if not node.pre]

    def connect_nodes(self):
        """ Connect all nodes registered """
        for node in self._nodes:
            node.set_pre(*[
                n for n in self._nodes
                if n.output_handle in node.input_handle])
            node.set_post(*[
                n for n in self._nodes
                if node.output_handle in n.input_handle])

    def get_node(self, module: snn_module.HXModule) -> Node:
        """
        Retrieve the node containing `module`.

        :param module: The module for which to find its corresponding node.

        :return: Returns the node containing `module`.
        """
        nodes = [n for n in self if n.module == module]
        assert len(nodes) == 1
        return nodes.pop()

    def module_exists(self, module: snn_module.HXModule) -> bool:
        """
        Checks whether a module is present in any of the nodes in `Modules`.

        :param module: The module to check if an node exists for.
        """
        return len([n for n in self if n.module == module]) > 0

    def pre_populations(self, node: List[Node]) -> Node:
        """
        Static method to traverse back in the graph to find the previous
        population relative to the module in `node`. This does only include
        nodes holding populations which have to be represented on hardware.
        Currently those are: [InputNeuron, Neuron, ReadoutNeuron].

        :param node: The node for which to find the pre-populations.

        :return: The list of pre-populations of the moduel in `node`.
        """
        # List of pre-nodes to check
        nodes = node.pre.copy()
        # List of actual pre-populations
        pre_pops = []

        while True:
            # No pre-nodes left
            if not nodes:
                return pre_pops

            # Get next node
            node = nodes.pop()

            # If of correct type its pre-node
            if isinstance(node.module, self._population_types):
                pre_pops.append(node)
                continue

            # Add next pre-nodes to check for population
            nodes += node.pre

    def post_populations(self, node: Node) -> Node:
        """
        Static method to traverse forward in the graph to find the next
        population relative to the module in `node`. This does only include
        nodes holding populations which have to be represented on hardware.
        Currently those are: [InputNeuron, Neuron, ReadoutNeuron].

        :param node: The node for which to find the post-populations.

        :return: The list of post-populations of the moduel in `node`.
        """
        # List of post-nodes to check
        nodes = node.post.copy()
        # List of actual post-populations
        post_pops = []

        while True:
            # No post-nodes left
            if not nodes:
                return post_pops

            # Get next node
            node = nodes.pop()

            # If population, return
            if isinstance(node.module, self._population_types):
                post_pops.append(node)
                continue

            # Add next post-nodes to check for population
            nodes += node.post


class ModuleManager(BaseModuleManager):

    """
    Object representing all nodes in a graph-like data structure.
    """

    # Types that are recognized as populations on hardware
    _population_types = (snn_module.InputNeuron, snn_module.Neuron,
                         snn_module.ReadoutNeuron)
    # Types that are recognized as projections on hardware
    _projection_types = (snn_module.Synapse,)

    def __init__(self) -> None:
        """
        Initialize a `Modules` object. This object holds a list of `nodes`.
        """
        super().__init__()
        # self._nodes: List[Node] = []
        self.populations: Set[Node] = set()
        self.projections: Set[Node] = set()
        self._input_populations: Dict[Node, Node] = {}

    def clear(self) -> None:
        """
        Override clear
        """
        super().clear()
        self.populations = set()
        self.projections = set()
        self._input_populations = {}

    def _set_dropout_mask(self) -> None:
        """
        Method to back-assign spiking mask from Dropout layers to the previous
        Neuron layer in order to configure the neurons appropriately on
        hardware.

        Note: BatchDropout layers have to be preceded by a Neuron layer.
        """
        for node in self._nodes:
            if isinstance(node.module, snn_module.BatchDropout):
                for pre in node.pre:
                    assert isinstance(pre.module, snn_module.Neuron)
                    pre.module.mask = node.module.set_mask()

    def _update_node(self, module: snn_module.HXModule,
                     input_handle: Union[Tuple[handle.TensorHandle],
                                         handle.TensorHandle],
                     output_handle: handle.TensorHandle) -> Node:
        """
        Update the input and output handle in an existing node in the `Modules`
        object.

        :param module: The module for whose node the handles are updated.
        :param input_handle: The new input handle, or a tuple of input handles.
        :param output_handle: The new output handle.

        :return: Returns the updated node.
        """
        if not isinstance(input_handle, tuple):
            input_handle = (input_handle,)
        node = self.get_node(module)
        node.set_handles(input_handle, output_handle)
        return node

    def _inject_input_nodes(self, instance) -> None:
        """
        On hardware, synapse have external populations as sources. Hence we
        have to augment Synapses with InputNeurons.
        """
        # Find input nodes which do not qualify as input
        output_handles = [n.output_handle for n in self._nodes]
        # Get all nodes that do not have an input handle that is an output
        # handel
        input_nodes = [
            n for n in self._nodes if isinstance(
                n.module, self._projection_types)  # pylint: disable=no-member
            and n not in self._input_populations
            and any(h not in output_handles for h in n.input_handle)]

        # Register nodes preceding those input nodes
        for node in input_nodes:
            # Create input layer and node
            source = snn_module.InputNeuron(node.module.in_features, instance)
            input_node = Node(source)
            self._nodes.append(input_node)
            self._input_populations.update({node: input_node})

    def _forward_input_nodes(self) -> None:
        """
        Forward modules that are registered as input populations. This is
        necessary since input populations are not created by the user but by
        the ModuleManger since they are needed for mapping the network to
        hardware. While all other modules are forwarded in the model created by
        the user, the input modules are not. Hence they are forwarded
        implicitly.
        """
        # Forward input populations and shift handles
        for post_node, in_node in self._input_populations.items():
            # Forward input node
            out_handle = in_node.module(*post_node.input_handle)
            # Shift handles for post module
            post_node.set_handles(out_handle, post_node.output_handle)

    def changed_since_last_run(self) -> bool:
        """
        Check if any module is marked dirty.

        :return: Returns true if at least one module has been marked dirty.
        """
        return any(node.module.changed_since_last_run for node in self._nodes)

    def reset_changed_since_last_run(self):
        """
        Restes all registered modules changed_since_last_run flags.
        """
        for node in self._nodes:
            node.module.reset_changed_since_last_run()

    def add(self, module: snn_module.HXModule,
            input_handles: Union[
                Tuple[handle.TensorHandle], handle.TensorHandle],
            output_handle: handle.TensorHandle) -> Node:
        """
        Adds a new node to the graph without creating edges. If the given
        module already exists, the input and output handles are updated (This
        is necessary since new handles are created with each forward-call).

        :param module: HXModule to add to the graph.
        :param input_handles: The input HXHandle (or a tuple of handles),
            passed to the modules forward method.
        :param output_handle: The output HXHandle returned by the modules
            forward method.

        :return: Returns the created/updates node.
        """
        if not isinstance(input_handles, tuple):
            input_handles = (input_handles,)
        if self.module_exists(module):
            node = self._update_node(module, input_handles, output_handle)
        else:
            node = Node(module, input_handles, output_handle)
            self._nodes.append(node)
        return node

    def pre_process(self, instance) -> None:
        """
        Inferres output sizes for each layer from its preceding layer and
        injects neuron masks in neuron layers and synapses in case of a
        subsequent dropout layer. Unlike in simulation, dropout masks need to
        be known before execution on hardware.

        :param instance: The instance to pre-process the modules for.
        """
        if not instance.mock:
            self._inject_input_nodes(instance)
            self._forward_input_nodes()

        # Create or update edges (in case graph changed)
        self.connect_nodes()

        # Set dropout masks
        self._set_dropout_mask()

    def ordered(self) -> List[Node]:
        """
        Create a list of nodes which can be looped in the correct order. This
        means for each node in the list its pre-nodes are at previous postions.

        :return: Returns the ordered list.
        """
        ordered_nodes = []

        # Nodes that are inputs
        input_nodes = self.inputs()
        nodes = [] + input_nodes

        while True:
            # No post nodes to check left
            if not nodes:
                return ordered_nodes

            # Next node
            node = nodes.pop(0)

            # If node already processed we can continue
            if node in ordered_nodes:
                continue

            # Otherwise this node need to be processed next
            ordered_nodes.append(node)

            # Append next nodes
            nodes += node.post
