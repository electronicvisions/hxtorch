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

import _hxtorch_core
import hxtorch.spiking.functional as F
from hxtorch.spiking.transforms import weight_transforms
from hxtorch.spiking.handle import LIFObservables, SynapseHandle
from hxtorch.spiking.modules.types import Projection
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment
    from hxtorch.spiking.execution_instance import ExecutionInstance
    from pyhalco_hicann_dls_vx_v3 import DLSGlobal

log = logger.get("hxtorch.spiking.modules")


class Synapse(Projection):  # pylint: disable=abstract-method
    """
    Synapse layer

    Caveat:
    For execution on hardware, this module can only be used in conjunction with
    a subsequent Neuron module.
    """

    output_type: Type = SynapseHandle

    # pylint: disable=too-many-arguments
    def __init__(self, in_features: int, out_features: int,
                 experiment: Experiment,
                 execution_instance: Optional[ExecutionInstance] = None,
                 chip_coordinate: Optional[DLSGlobal] = None,
                 device: str = None, dtype: Type = None,
                 transform: Callable = weight_transforms.linear_saturating) \
            -> None:
        """
        TODO: Think about what to do with device here.

        :param in_features: Size of input dimension.
        :param out_features: Size of output dimension.
        :param device: Device to execute on. Only considered in mock-mode.
        :param dtype: Data type of weight tensor.
        :param experiment: Experiment to append layer to.
        :param execution_instance: Execution instance to place to.
        :param chip_coordinate: Chip coordinate this module is placed on.
        """
        super().__init__(
            in_features, out_features, experiment=experiment,
            execution_instance=execution_instance,
            chip_coordinate=chip_coordinate,
        )

        self.weight = Parameter(
            torch.empty(
                (out_features, in_features), device=device, dtype=dtype))
        self._weight_old = torch.zeros_like(self.weight.data)
        self.weight_transform = transform

        self.reset_parameters()

    @property
    def changed_since_last_run(self) -> bool:
        """
        Getter for changed_since_last_run.

        :returns: Boolean indicating wether module changed since last run.
        """
        self._weight_old.to(self.weight.data.device)
        return not torch.equal(self.weight.data,
                               self._weight_old.to(self.weight.data.device)) \
            or self._changed_since_last_run

    def reset_changed_since_last_run(self) -> None:
        """
        Reset changed_since_last_run. Sets the corresponding flag to false.
        """
        self._weight_old = torch.clone(self.weight.data)
        return super().reset_changed_since_last_run()

    def reset_parameters(self) -> None:
        """
        Resets the synapses weights by reinitialization using
        `torch.nn.kaiming_uniform_`.
        """
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def register_hw_entity(self) -> None:
        """
        Add the synapse layer to the experiment's projections.
        """
        self.experiment.register_projection(self)

    # pylint: disable=too-many-locals
    def add_to_network_graph(
            self,
            pre: grenade.network.PopulationOnNetwork,
            post: grenade.network.PopulationOnNetwork,
            builder: grenade.network.NetworkBuilder) -> Tuple[
                grenade.network.ProjectionOnNetwork, ...]:
        """
        Adds the projection to a grenade network builder by providing the
        population descriptor of the corresponding pre and post population.
        Note: This creates one inhibitory and one excitatory projection on
        hardware in order to represent signed hardware weights.

        :param pre: Population descriptor of pre-population.
        :param post: Population descriptor of post-population.
        :param builder: Grenade network builder to add projection to.

        :returns: A tuple of grenade ProjectionOnNetworks holding the
            descriptors for the excitatory and inhibitory projection.
        """
        weight_transformed = self.weight_transform(
            torch.clone(self.weight.data))

        weight_exc = torch.clone(weight_transformed)
        weight_inh = torch.clone(weight_transformed)
        weight_exc[weight_exc < 0.] = 0
        weight_inh[weight_inh >= 0.] = 0

        weight_exc += .5
        weight_inh -= .5

        # TODO: Make sure this doesn't require rerouting
        connections_exc = _hxtorch_core.weight_to_connection(  # pylint: disable=no-member
            weight_exc.int().cpu().numpy())
        connections_inh = _hxtorch_core.weight_to_connection(  # pylint: disable=no-member
            weight_inh.int().cpu().numpy())

        # add inter-execution-instance projection and source if necessary
        if pre.toExecutionInstanceID() != post.toExecutionInstanceID():
            pre_size = weight_transformed.shape[1]

            iei_pre = builder.add(grenade.network.ExternalSourcePopulation(
                [grenade.network.ExternalSourcePopulation.Neuron(False)]
                * pre_size, self.chip_coordinate),
                self.execution_instance.ID)

            # [nrn on pop pre, compartment on nrn pre,
            #  nrn on pop post, compartment on nrn post, delay in clock cycles]
            connections = np.array([[i, 0, i, 0, 0] for i in range(pre_size)])
            iei_projection = grenade.network.InterExecutionInstanceProjection()
            iei_projection.from_numpy(connections, pre, iei_pre)

            builder.add(iei_projection)

            pre = iei_pre

        projection_exc = grenade.network.Projection(
            grenade.network.Receptor(
                grenade.network.Receptor.ID(),
                grenade.network.Receptor.Type.excitatory),
            connections_exc, pre, post, self.chip_coordinate)
        projection_inh = grenade.network.Projection(
            grenade.network.Receptor(
                grenade.network.Receptor.ID(),
                grenade.network.Receptor.Type.inhibitory),
            connections_inh, pre, post, self.chip_coordinate)

        exc_descriptor = builder.add(
            projection_exc, self.execution_instance.ID)
        inh_descriptor = builder.add(
            projection_inh, self.execution_instance.ID)
        self.descriptor = (exc_descriptor, inh_descriptor)
        log.TRACE(f"Added projection '{self}' to grenade graph.")

        return self.descriptor

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
