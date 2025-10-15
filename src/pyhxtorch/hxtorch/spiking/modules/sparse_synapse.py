"""
Implementing SNN modules
"""
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Type,
    Optional,
    Tuple,
)
import math

import torch
from torch.nn.parameter import Parameter

import pygrenade_vx as grenade

from hxtorch.core.modules.projection import ProjectionConnection
import hxtorch.spiking.functional as F
from hxtorch.spiking.transforms import weight_transforms
from hxtorch.spiking.handle import LIFObservables, SynapseHandle
from hxtorch.spiking.modules.types.projection import Projection
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment


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
                 chip_coordinate: Optional[
                     Tuple[grenade.common.ChipOnConnection,
                           grenade.common.ConnectionOnExecutor]] = None,
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
        :param chip_coordinate: Chip coordinate this module is placed on.
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
            chip_coordinate=chip_coordinate)

        self.bias = None
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

    def reset_parameters(self) -> None:
        """
        Resets the synapses weights by reinitialization using
        `torch.nn.kaiming_uniform_`.
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data[~self.mask] = 0.

    def get_connections(self) -> List[grenade.network.Connection]:
        # TODO: Use sparse weight parameters in the future
        weight = self.weight.detach()[self.connections]
        weight_transformed = self.weight_transform(weight)

        connections = [
            ProjectionConnection(col, row, weight)
            for (row, col, weight) in zip(
                *self.connections,
                weight_transformed.round().cpu().numpy(),
            )
        ]

        return connections

    # pylint: disable=redefined-builtin, arguments-differ
    def forward_func(self, input: LIFObservables) -> SynapseHandle:
        return SynapseHandle(F.linear_sparse(
            input.spikes, self.weight, self.mask, None))
