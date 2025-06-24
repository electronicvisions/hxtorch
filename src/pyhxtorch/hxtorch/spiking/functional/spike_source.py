"""
Define different input spike sources
"""
from typing import Optional
import torch

from hxtorch.spiking.handle import LIFObservables
from hxtorch.spiking.functional.unterjubel import Unterjubel


# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin
def input_neuron(input: torch.Tensor,
                 hw_data: Optional[torch.Tensor] = None) -> LIFObservables:
    """
    Input neuron, forwards spikes without modification in non-hardware runs
    but injects loop-back recorded spikes if available.

    :param input: Input spike tensor.
    :param hw_data: Loop-back spikes, if available.

    :returns: Returns the input spike tensor.
    """
    if hw_data is None:
        return input

    time_steps, _, _ = input.shape
    return Unterjubel.apply(input, hw_data.to(input.device)[:time_steps, ...])
