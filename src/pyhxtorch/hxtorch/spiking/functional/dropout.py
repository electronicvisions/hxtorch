"""
Custom BatchDropout function
"""
import torch


# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin
def batch_dropout(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a dropout mask to a batch of inputs.

    :param input: The input tensor to apply dropout to.
    :param mask: The dropout mask. Entires in the mask which are `False` will
                 disable their corresponding entry in `input`.

    :returns: The input tensor with dropout mask applied.
    """
    return input * mask.to(input.device)
