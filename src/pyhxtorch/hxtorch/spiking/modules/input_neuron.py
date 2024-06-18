"""
Implementing input neuron module
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Type, Tuple, Optional
import pylogging as logger

import torch

import pygrenade_vx as grenade

from _hxtorch_spiking import tensor_to_spike_times  # pylint: disable=import-error
import hxtorch.spiking.functional as F
from hxtorch.spiking.handle import NeuronHandle
from hxtorch.spiking.modules.types import InputPopulation

if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment
    from hxtorch.spiking.observables import HardwareObservables
    from hxtorch.spiking.execution_instance import ExecutionInstance

log = logger.get("hxtorch.spiking.modules")


class InputNeuron(InputPopulation):
    """
    Spike source generating spikes at the times [ms] given in the spike_times
    array.
    """
    output_type: Type = NeuronHandle

    def __init__(
            self, size: int, experiment: Experiment,
            execution_instance: Optional[ExecutionInstance] = None) -> None:
        """
        Instantiate a InputNeuron. This module serves as an External
        Population for input injection and is created within `experiment`
        if not present in the considered model.
        This module performs an identity mapping when `forward` is called.

        :param size: Number of input neurons.
        :param experiment: Experiment to which this module is assigned.
        :param execution_instance: Execution instance to place to.
        """
        super().__init__(
            size, experiment, execution_instance=execution_instance)

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
        gpopulation = grenade.network.ExternalSourcePopulation(
            [grenade.network.ExternalSourcePopulation.Neuron(
                self.execution_instance.input_loopback)] * self.size)
        # add to builder
        self.descriptor = builder.add(gpopulation, self.execution_instance.ID)
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

    def post_process(self, hw_data: HardwareObservables, runtime: float) \
            -> Tuple[Optional[torch.Tensor], ...]:
        if self.execution_instance.input_loopback:
            return hw_data.spikes.to_dense(runtime, self.experiment.dt).float()

        return None

    # pylint: disable=redefined-builtin
    def forward_func(self, input: NeuronHandle,
                     hw_data: Optional[Tuple[torch.Tensor]] = None) \
            -> NeuronHandle:
        return NeuronHandle(
            spikes=F.input_neuron(input.spikes, hw_data=hw_data))
