"""
Implementing SNN modules
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Tuple, Type, Optional
import math
import numpy as np
import pylogging as logger

import torch
from torch.nn.parameter import Parameter

import pygrenade_vx as grenade

from _hxtorch_core import weight_to_connection
import hxtorch.spiking.functional as F
from hxtorch.spiking.transforms import weight_transforms
from hxtorch.spiking.handle import NeuronHandle, SynapseHandle
from hxtorch.spiking.modules.synapse import Projection
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment
    from hxtorch.spiking.execution_instance import ExecutionInstance

log = logger.get("hxtorch.spiking.modules")


class SparseSynapse(Projection):  # pylint: disable=abstract-method
    """
    Sparse synapse layer

    Caveat:
    For execution on hardware, this module can only be used in conjuction with
    a subsequent Neuron module.
    """
    __constants__ = ['connections', 'in_features', 'out_features']
    connections: torch.Tensor
    output_type: Type = SynapseHandle

    # pylint: disable=too-many-arguments
    def __init__(self, connections: torch.SparseTensor,
                 experiment: Experiment,
                 execution_instance: Optional[ExecutionInstance] = None,
                 device: str = None, dtype: Type = None,
                 transform: Callable = weight_transforms.linear_saturating) \
            -> None:
        """
        A sparse projection, with connections defined by non-zero entries in
        `connections`, represented sparsely on hardware.

        :param connections: A tensor of shape (in_features, out_features)
            defining existing connections by one-entries. Can be sparse or
            non-sparse.
        :param experiment: Experiment to append layer to.
        :param device: Device to execute on. Only considered in mock-mode.
        :param dtype: Data type of weight tensor.
        :param transform: A function taking the modules weight tensor and
            transforms it into weights mappable to hardware.
        """
        # TODO: Backend needs to know about projection. Find solution so mark
        # projections modules properly
        connections = connections.transpose(1, 0)
        if not connections.is_sparse:
            connections = connections.to_sparse()
        if not connections.is_coalesced():
            connections = connections.coalesce()
        self.connections = connections.indices().tolist()
        self.mask = connections.bool().to_dense().to(device)

        super().__init__(
            self.mask.shape[1], self.mask.shape[0], experiment=experiment,
            execution_instance=execution_instance)

        self.weight = Parameter(
            torch.empty(
                (self.out_features, self.in_features), device=device,
                dtype=dtype))
        self._weight_old = torch.zeros_like(self.weight.data, device=device)
        self.weight_transform = transform

        self.reset_parameters()

    def extra_repr(self) -> str:
        """ Add additional information """
        return f"number connections={len(self.connections[0])}, " \
            + f"{super().extra_repr()}"

    @property
    def changed_since_last_run(self) -> bool:
        """
        Getter for changed_since_last_run.

        :returns: Boolean indicating wether module changed since last run.
        """
        return not torch.equal(self.weight.data, self._weight_old) \
            or self._changed_since_last_run

    def reset_changed_since_last_run(self) -> None:
        """
        Reset changed_since_last_run. Sets the corresponding flag to false.
        """
        self._weight_old = torch.clone(self.weight.data)
        return super().reset_changed_since_last_run()

    def reset_parameters(self) -> None:
        """
        Resets the synapses weights by reinitalization using
        `torch.nn.kaiming_uniform_`.
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data[~self.mask] = 0.

    def register_hw_entity(self) -> None:
        """
        Add the synapse layer to the experiment's projections.
        """
        self.experiment.register_projection(self)

    # pylint: disable=too-many-locals
    def add_to_network_graph(
            self,
            pre: grenade.PopulationDescriptor,
            post: grenade.PopulationDescriptor,
            builder: grenade.NetworkBuilder) -> Tuple[
                grenade.ProjectionDescriptor, ...]:
        """
        Adds the projection to a grenade network builder by providing the
        population descriptor of the corresponding pre and post population.
        Note: This creates one inhibitory and one excitatory population on
        hardware in order to represent signed hardware weights.

        :param pre: Population descriptor of pre-population.
        :param post: Population descriptor of post-population.
        :param builder: Greande netowrk builder to add projection to.

        :returns: A tuple of grenade ProjectionDescriptors holding the
            descriptors for the excitatory and inhibitory projection.
        """
        # TODO: Use sparse weight parameters in the future
        weight = self.weight.detach()[self.connections]
        weight_transformed = self.weight_transform(weight)

        weight_exc = torch.clone(weight_transformed)
        weight_inh = torch.clone(weight_transformed)
        weight_exc[weight_exc < 0.] = 0
        weight_inh[weight_inh >= 0.] = 0

        # TODO: Make sure this doesn't require rerouting
        connections_exc = weight_to_connection(weight_exc, self.connections)  # pylint: disable=no-member
        connections_inh = weight_to_connection(weight_inh, self.connections)  # pylint: disable=no-member

        # add inter-execution-instance projection and source if necessary
        # TODO: Move this to `Experiment`.`
        if pre.toExecutionInstanceID() != post.toExecutionInstanceID():
            iei_pre = builder.add(grenade.network.ExternalSourcePopulation(
                [grenade.network.ExternalSourcePopulation.Neuron(False)]
                * self.in_features),
                self.execution_instance)

            # [nrn on pop pre, compartment on nrn pre,
            #  nrn on pop post, compartment on nrn post]
            connections = np.array(
                [[i, 0, i, 0] for i in range(self.in_features)])
            iei_projection = grenade.network.InterExecutionInstanceProjection()
            iei_projection.from_numpy(connections, pre, iei_pre)

            builder.add(iei_projection)

            pre = iei_pre

        projection_exc = grenade.network.Projection(
            grenade.network.Receptor(
                grenade.network.Receptor.ID(),
                grenade.network.Receptor.Type.excitatory),
            connections_exc, pre, post)
        projection_inh = grenade.network.Projection(
            grenade.network.Receptor(
                grenade.network.Receptor.ID(),
                grenade.network.Receptor.Type.inhibitory),
            connections_inh, pre, post)

        exc_descriptor = builder.add(
            projection_exc, self.execution_instance.ID)
        inh_descriptor = builder.add(
            projection_inh, self.execution_instance.ID)
        self.descriptor = (exc_descriptor, inh_descriptor)
        log.TRACE(f"Added sparse projection '{self}' to grenade graph.")

        return self.descriptor

    # pylint: disable=redefined-builtin, arguments-differ
    def forward_func(self, input: NeuronHandle) -> SynapseHandle:
        return SynapseHandle(F.linear_sparse(
            input.spikes, self.weight, self.mask, None))
