"""
Define torch.autograd.Function to mock discretization (e.g. weight
discretization).
"""
# pylint: disable=redefined-builtin, arguments-differ, abstract-method
from typing import Tuple, Optional
import torch


class Discrete(torch.autograd.Function):
    """
    Discretize values of a tensor.
    """

    # pylint: disable=unused-argument
    @staticmethod
    def forward(ctx, input: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """
        Rounds values of `input` to closest multiple of `step`.
        """
        return torch.round(input / step) * step

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) \
            -> Tuple[Optional[torch.Tensor]]:
        grad_input = grad_step = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        return grad_input, grad_step
