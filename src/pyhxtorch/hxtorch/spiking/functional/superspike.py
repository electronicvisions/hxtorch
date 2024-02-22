"""
Surrograte gradient for SuperSpike.
Basically reimplemented from:
https://github.com/norse/norse/blob/18cd8aa256c3e7b5a28a852f5e928b104bc637fc/norse/torch/functional/superspike.py#L33
"""
from typing import Optional, Tuple
import torch


# Issue: 4019
# pylint: disable=abstract-method
class SuperSpike(torch.autograd.Function):
    """
    Define Surrogate Gradient 'SuperSpike' (negative side of Fast Sigmoid)
    See: https://arxiv.org/abs/1705.11146
    """

    # Allow redefining builtin for PyTorch consistency
    # pylint: disable=redefined-builtin, arguments-differ
    @staticmethod
    def forward(ctx, input: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Forward function, generating spikes at positions > 0.

        :param input: Tensor holding `v_mem - v_th`, of > 0 a spike at the
            corresponding entry is generated. Saved for backward.
        :param alpha: Parameter controlling the slope of the surrogate
            derivative. Saved for backward.

        :returns: A boolean tensor holding True at the entries for neurons
            emitting a spike.
        """
        ctx.save_for_backward(input)
        # Alpha for backward
        ctx.alpha = alpha

        return torch.gt(input, 0.0).to(input.dtype)

    # Allow redefining builtin for PyTorch consistency
    # pylint: disable=redefined-builtin
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) \
            -> Tuple[Optional[torch.Tensor], ...]:
        """
        Implements surrogate gradient 'SuperSpike' for backward.

        :param grad_output: Back-propagated gradient.

        :returns: SuperSpike gradient multiplied to back-propagated gradient.
        """
        input, = ctx.saved_tensors
        grad = grad_output.clone()
        grad_input = grad / (ctx.alpha * torch.abs(input) + 1.0).pow(2)

        return grad_input, None


# pylint: disable=redefined-builtin, arguments-differ
def superspike_func(input: torch.Tensor, alpha: float) -> torch.Tensor:
    return SuperSpike.apply(input, alpha)
