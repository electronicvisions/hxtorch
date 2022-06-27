"""
Define Nodes and a Graph-based datastructure holding Nodes.
"""
from typing import Optional, Tuple, List, Union

import pygrenade_vx as grenade
import hxtorch.snn.modules as snn_module
import hxtorch.snn.handle as handle


class Node:

    """
    Node class for nodes of high-level network graph, needed before hardware
    execution, allowing retrieving information for mapping the experiemnt onto
    hardware.
    """

    def __init__(self, module: snn_module.HXModule,
                 input_handle: Optional[Tuple[handle.TensorHandle]] = None,
                 output_handle: Optional[handle.TensorHandle] = None) \
            -> None:
        """
        Instantiate a Node.

        :param module: The (HX) module represented by this node.
        :param input_handle: The input handle connecting the pre-module to
            `module`.
        :param output_handle: The output handle connecting `module` to the
            post-module.
        """
        self.module = module
        # TODO: Maybe the handles can be moved to HXModule
        self.input_handle = input_handle
        self.output_handle = output_handle
        self.pre: List['Node'] = list()
        self.post: List['Node'] = list()
        self.descriptor: Union[
            grenade.PopulationDescriptor,
            Tuple[grenade.ProjectionDescriptor]] = None

    def __str__(self):
        pre_nodes = [f"Node(at {hex(id(node))})" for node in self.pre]
        post_nodes = [f"Node(at {hex(id(node))})" for node in self.post]
        string = f"Node(at {hex(id(self))}):\n" \
            + f"\tModule: {self.module} (type: {type(self.module)}, " \
            + f"at: {hex(id(self.module))})\n" \
            + f"\tInput: {self.input_handle}\n" \
            + f"\tOutput: {self.output_handle}\n" \
            + f"\tPre-Nodes: {pre_nodes}\n" \
            + f"\tPost-Nodes: {post_nodes}\n" \
            + f"\tGrenade Descriptor: {self.descriptor}"
        return string

    def set_handles(self, input_handles: Tuple[handle.TensorHandle],
                    output_handle: handle.TensorHandle) -> None:
        """
        Assign input and ouput handles to the node.

        :param input_handles: Tuple of input input handles.
        :param output_handle: Output handle.
        """
        self.input_handle = (input_handles,) \
            if not isinstance(input_handles, tuple) else input_handles
        self.output_handle = output_handle

    def set_post(self, *nodes: Tuple["Node"]):
        """
        Set post populations.

        :param nodes: Nodes to assign as post-nodes.
        """
        self.post = list(nodes)

    def set_pre(self, *nodes: Tuple["Node"]):
        """
        Set prepopulations

        :param nodes: Nodes to assign as pre-nodes.
        """
        self.pre = list(nodes)
