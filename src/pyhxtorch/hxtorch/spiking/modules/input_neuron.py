"""
Implementing input neuron module
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Type, Optional
import pylogging as logger

import torch

import pygrenade_vx.network.placed_logical as grenade

from _hxtorch_spiking import tensor_to_spike_times, DataHandle  # pylint: disable=import-error
import hxtorch.spiking.functional as F
from hxtorch.spiking.handle import NeuronHandle
from hxtorch.spiking.modules.hx_module import HXModule
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment

log = logger.get("hxtorch.spiking.modules")


class InputNeuron(HXModule):
    """
    Spike source generating spikes at the times [ms] given in the spike_times
    array.
    """
    output_type: Type = NeuronHandle

    def __init__(self, size: int, experiment: Experiment) -> None:
        """
        Instanziate a InputNeuron. This module serves as an External
        Population for input injection and is created within `experiment`
        if not present in the considerd model.
        This module performes an identity mapping when `forward` is called.

        :param size: Number of input neurons.
        :param experiment: Experiment to which this module is assigned.
        """
        super().__init__(experiment, func=F.input_neuron)
        self.size = size

    def register_hw_entity(self) -> None:
        """
        Register instance in member `experiment`.
        """
        self.experiment.register_population(self)

    def add_to_network_graph(
        self, builder: grenade.NetworkBuilder) \
            -> grenade.PopulationDescriptor:
        """
        Adds instance to grenade's network builder.

        :param builder: Grenade network builder to add extrenal population to.
        :returns: External population descriptor.
        """
        # create grenade population
        gpopulation = grenade.ExternalPopulation(self.size)
        # add to builder
        self.descriptor = builder.add(gpopulation)
        log.TRACE(f"Added Input Population: {self}")

        return self.descriptor

    def add_to_input_generator(
            self, input: NeuronHandle,  # pylint: disable=redefined-builtin
            builder: grenade.InputGenerator) -> None:
        """
        Add the neurons events represented by this instance to grenades input
        generator.

        :param input: Dense spike tensor. These spikes are implictely converted
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
            input.spikes, dt=self.experiment.dt / 1e-3)
        builder.add(spike_times, self.descriptor)

    def post_process(self, hw_spikes: Optional[DataHandle],
                     hw_cadc: Optional[DataHandle],
                     hw_madc: Optional[DataHandle]) \
            -> Tuple[Optional[torch.Tensor], ...]:
        pass
