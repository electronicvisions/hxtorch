"""
Implementing SNN modules
"""
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Callable,
    Type,
    Optional,
    List,
    Tuple,
    Literal,
)
import math

import numpy as np
import torch
from torch.nn.parameter import Parameter

import pygrenade_vx as grenade


import hxtorch.spiking.functional as F
from hxtorch.core.plasticity_rule import PlasticityRule
from hxtorch.core.modules.projection import ProjectionConnection
from hxtorch.spiking.transforms import weight_transforms
from hxtorch.spiking.handle import LIFObservables, SynapseHandle
from hxtorch.spiking.modules.types.projection import Projection

if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment


class Synapse(Projection):  # pylint: disable=abstract-method
    """
    Synapse layer

    Caveat:
    For execution on hardware, this module can only be used in conjunction with
    a subsequent Neuron module.
    """

    output_type: Type = SynapseHandle

    # pylint: disable=too-many-arguments
    def __init__(
        self, in_features: int, out_features: int,
        experiment: Experiment,
        chip_coordinate: Optional[
            Tuple[grenade.common.ChipOnConnection,
                  grenade.common.ConnectionOnExecutor]] = None,
        device: str = None,
        dtype: Type = None,
        transform: Callable = weight_transforms.linear_saturating,
        plasticity_rule: PlasticityRule | None = None,
        receptor: Literal["excitatory", "inhibitory"]
            | List[str] | Tuple[str, ...] | None = None,
    ) -> None:
        """
        TODO: Think about what to do with device here.

        :param in_features: Size of input dimension.
        :param out_features: Size of output dimension.
        :param experiment: Experiment to append layer to.
        :param chip_coordinate: Chip coordinate this module is placed on.
        :param device: Device to execute on. Only considered in mock-mode.
        :param dtype: Data type of weight tensor.
        :param plasticity_rule: Plasticity rule adjusting this synapse.
        :param: receptor: Receptor type of the synapse. Can be 'excitatory',
            'inhibitory' or ('excitatory', inhibitory') for a signed synapse.
        """
        super().__init__(
            in_features,
            out_features,
            experiment=experiment,
            chip_coordinate=chip_coordinate,
            plasticity_rule=plasticity_rule,
            receptor=receptor,
        )

        self.weight = Parameter(
            torch.empty(
                (out_features, in_features), device=device, dtype=dtype))
        self.weight_transform = transform

        self._weight_hash = None

        self.reset_parameters()

    @property
    def changed_input_data(self) -> bool:
        """
        Getter for changed_since_last_run.

        :returns: Boolean indicating wether module changed since last run.
        """
        if self._weight_hash is None:
            return True
        return not hash(self.weight.data) == self._weight_hash

    @changed_input_data.setter
    # pylint: disable=unused-argument
    def changed_input_data(self, changed: bool) -> bool:
        if hasattr(self, "weight"):
            self._weight_hash = hash(self.weight.data)

    def reset_parameters(self) -> None:
        """
        Resets the synapses weights by reinitialization using
        `torch.nn.kaiming_uniform_`.
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def get_connections(self) -> List[Tuple[int, int, float]]:
        weight_transformed = self.weight_transform(
            torch.clone(self.weight.data))
        # TODO: Make sure this doesn't require rerouting
        connections = [
            ProjectionConnection(col, row, weight)
            for (row, col), weight in np.ndenumerate(
                weight_transformed.round().int().cpu().numpy()
            )
        ]
        return connections

    # pylint: disable=redefined-builtin, arguments-differ
    def forward_func(self, input: LIFObservables) -> SynapseHandle:
        return SynapseHandle(
            graded_spikes=F.linear(input.spikes, self.weight, None))


class EventPropSynapse(Synapse):
    # pylint: disable=redefined-builtin, arguments-differ
    def forward_func(self, input: LIFObservables) -> SynapseHandle:
        return SynapseHandle(
            graded_spikes=F.EventPropSynapseFunction.apply(
                input.spikes, self.weight))
