"""
Custom BatchDropout function
"""
import torch

from hxtorch.spiking.handle import NeuronHandle


# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin
def batch_dropout(input: NeuronHandle, mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a dropout mask to a batch of inputs.

    :param input: The input tensor handle to apply dropout to.
    :param mask: The dropout mask. Entires in the mask which are `False` will
                 disable their corresponding entry in `input`.

    :returns: The input tensor with dropout mask applied.
    """
    return NeuronHandle(spikes=input.spikes * mask.to(input.device))
