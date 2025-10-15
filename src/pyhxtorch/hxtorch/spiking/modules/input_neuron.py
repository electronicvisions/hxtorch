"""
Implementing input neuron module
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Type, Tuple, Optional

import torch

from _hxtorch_spiking import tensor_to_spike_times  # pylint: disable=import-error
import hxtorch.spiking.functional as F
from hxtorch.spiking.handle import LIFObservables
from hxtorch.spiking.modules.types.population import InputPopulation

if TYPE_CHECKING:
    from hxtorch.spiking.observables import HXTorchObservables


class InputNeuron(InputPopulation):
    """
    Spike source generating spikes at the times given in the dense spike_times
    array binned with the time step of the experiment.
    """
    output_type: Type = LIFObservables

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._input_handle: Optional[LIFObservables] = None
        self._input_data_hash: Optional[int] = None

    @property
    def changed_input_data(self) -> bool:
        old_hash = self._input_data_hash
        self._input_handle = self.experiment.get_source_handle(self)
        self._input_data_hash = hash(self._input_handle.spikes)
        return old_hash != self._input_data_hash

    @changed_input_data.setter
    def changed_input_data(self, value: bool) -> None:
        pass

    def get_spike_times(self):
        # tensor to spike times
        # maybe support sparse input tensor?
        spike_times = tensor_to_spike_times(  # pylint: disable=no-member
            self._input_handle.spikes.cpu(),
            dt=self.experiment.dt,
        )
        self.changed_input_data = True
        return spike_times

    def post_process(
        self,
        hw_data: HXTorchObservables,
        runtime: float,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        if self.enable_spike_loopback:
            return hw_data.spikes.to_dense(runtime, self.experiment.dt).float()
        return None

    # pylint: disable=redefined-builtin
    def forward_func(
        self,
        input: LIFObservables,
        hw_data: Optional[Tuple[torch.Tensor]] = None,
    ) -> LIFObservables:
        return LIFObservables(
            spikes=F.input_neuron(input.spikes, hw_data=hw_data),
        )
