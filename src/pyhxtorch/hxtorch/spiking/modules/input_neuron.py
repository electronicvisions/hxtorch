"""
Implementing input neuron module
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Type, Optional
import pylogging as logger

import torch

import pygrenade_vx as grenade

from _hxtorch_spiking import (
    tensor_to_spike_times, SpikeHandle, CADCHandle, MADCHandle)  # pylint: disable=import-error
import hxtorch.spiking.functional as F
from hxtorch.spiking.handle import NeuronHandle
from hxtorch.spiking.modules.hx_module import HXModule
from hxtorch.spiking.modules.entity_on_execution_instance import \
    EntityOnExecutionInstance
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment

log = logger.get("hxtorch.spiking.modules")


class InputNeuron(HXModule, EntityOnExecutionInstance):
    """
    Spike source generating spikes at the times [ms] given in the spike_times
    array.
    """
    output_type: Type = NeuronHandle

    def __init__(
            self, size: int,
            experiment: Experiment,
            execution_instance: grenade.common.ExecutionInstanceID
            = grenade.common.ExecutionInstanceID()) -> None:
        """
        Instantiate a InputNeuron. This module serves as an External
        Population for input injection and is created within `experiment`
        if not present in the considered model.
        This module performs an identity mapping when `forward` is called.

        :param size: Number of input neurons.
        :param experiment: Experiment to which this module is assigned.
        :param execution_instance: Execution instance to place to.
        """
        HXModule.__init__(self, experiment, func=F.input_neuron)
        EntityOnExecutionInstance.__init__(self, execution_instance)
        self.size = size

    def extra_repr(self) -> str:
        """ Add additional information """
        reprs = f"execution_instance={self.execution_instance}, "
        reprs += f"size={self.size}, {super().extra_repr()}"
        return reprs

    def register_hw_entity(self) -> None:
        """
        Register instance in member `experiment`.
        """
        self.experiment.register_population(self)

    def add_to_network_graph(
        self, builder: grenade.network.NetworkBuilder) \
            -> grenade.network.PopulationOnNetwork:
        """
        Adds instance to grenade's network builder.

        :param builder: Grenade network builder to add external population to.
        :returns: External population descriptor.
        """
        # create grenade population
        gpopulation = grenade.network.ExternalSourcePopulation(self.size)
        # add to builder
        self.descriptor = builder.add(
            gpopulation, self.execution_instance)
        log.TRACE(f"Added Input Population: {self}")

        return self.descriptor

    def add_to_input_generator(
            self, input: NeuronHandle,  # pylint: disable=redefined-builtin
            builder: grenade.network.InputGenerator) -> None:
        """
        Add the neurons events represented by this instance to grenades input
        generator.

        :param input: Dense spike tensor. These spikes are implicitly converted
            to spike times. TODO: Allow sparse tensors.
        :param builder: Grenade's input generator to append the events to.
        """
        if isinstance(input, tuple):
            assert len(input) == 1
            input, = input  # unpack

        # tensor to spike times
        # maybe support sparse input tensor?
        # TODO: Expects ms relative. Align to time handling.
        spike_times = tensor_to_spike_times(  # pylint: disable=no-member
            input.spikes.cpu(), dt=self.experiment.dt / 1e-3)
        builder.add(spike_times, self.descriptor)

    def post_process(self, hw_spikes: Optional[SpikeHandle],
                     hw_cadc: Optional[CADCHandle],
                     hw_madc: Optional[MADCHandle],
                     runtime: float) -> Tuple[Optional[torch.Tensor], ...]:
        pass
