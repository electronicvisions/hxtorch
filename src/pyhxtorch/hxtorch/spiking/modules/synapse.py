"""
Implementing SNN modules
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Tuple, Type, Union
import numpy as np
import pylogging as logger

import torch
from torch.nn.parameter import Parameter

import pygrenade_vx.network as grenade

from _hxtorch_spiking import weight_to_connection
import hxtorch.spiking.functional as F
from hxtorch.spiking.transforms import weight_transforms
from hxtorch.spiking.handle import SynapseHandle
from hxtorch.spiking.modules.types import Projection
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment

log = logger.get("hxtorch.spiking.modules")


class Synapse(Projection):  # pylint: disable=abstract-method
    """
    Synapse layer

    Caveat:
    For execution on hardware, this module can only be used in conjuction with
    a subsequent Neuron module.
    """

    output_type: Type = SynapseHandle

    # pylint: disable=too-many-arguments
    def __init__(self, in_features: int, out_features: int,
                 experiment: Experiment,
                 func: Union[Callable, torch.autograd.Function] = F.linear,
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
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation. Required function args:
                [input (torch.Tensor), weight (torch.Tensor)]
        """
        super().__init__(experiment=experiment, func=func)

        self.in_features = in_features
        self.out_features = out_features
        self.size = out_features

        self.weight = Parameter(
            torch.empty(
                (out_features, in_features), device=device, dtype=dtype))
        self._weight_old = torch.zeros_like(self.weight.data)
        self.weight_transform = transform

        self.reset_parameters(1.0e-3, 1. / np.sqrt(in_features))
        self.extra_args = (self.weight, None)  # No bias

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

    def reset_parameters(self, mean: float, std: float) -> None:
        """
        Resets the synapses weights by reinitalization using torch.nn.normal_
        with mean=`mean` and std=`std`.

        :param mean: The mean of the normal distribution used to initialize the
            weights from.
        :param std: The standard deviation of the normal distribution used to
            initialized the weights from.
        """
        torch.nn.init.normal_(self.weight, mean=mean, std=std)

    def register_hw_entity(self) -> None:
        """
        Add the synapse layer to the experiment's projections.
        """
        self.experiment.register_projection(self)

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
        weight_transformed = self.weight_transform(
            torch.clone(self.weight.data))

        weight_exc = torch.clone(weight_transformed)
        weight_inh = torch.clone(weight_transformed)
        weight_exc[weight_exc < 0.] = 0
        weight_inh[weight_inh >= 0.] = 0

        # TODO: Make sure this doesn't require rerouting
        connections_exc = weight_to_connection(weight_exc)  # pylint: disable=no-member
        connections_inh = weight_to_connection(weight_inh)  # pylint: disable=no-member

        projection_exc = grenade.Projection(
            grenade.Receptor(
                grenade.Receptor.ID(),
                grenade.Receptor.Type.excitatory),
            connections_exc, pre, post)
        projection_inh = grenade.Projection(
            grenade.Receptor(
                grenade.Receptor.ID(),
                grenade.Receptor.Type.inhibitory),
            connections_inh, pre, post)

        exc_descriptor = builder.add(projection_exc)
        inh_descriptor = builder.add(projection_inh)
        self.descriptor = (exc_descriptor, inh_descriptor)
        log.TRACE(f"Added projection '{self}' to grenade graph.")

        return self.descriptor
