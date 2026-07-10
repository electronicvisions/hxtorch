"""
Define torch.autograd.Functions for dealing with noisy parameters (with
respect to fixed-pattern-noise).
"""
# pylint: disable=redefined-builtin, arguments-differ, abstract-method
from typing import Optional
import torch


class FixedPatternNoiseMultiply(torch.autograd.Function):
    """
    Multiply noiy factors to tensor. The noisy factors must be leave nodes in
    the compute graph. In the backwards-pass, their mean value (of the
    underlying distribution they are sampled from) is injected.
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, factor: torch.Tensor,
                mean: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mean is not None:
            ctx.save_for_backward(mean)
        else:
            ctx.save_for_backward(factor)
        return input * factor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_factor = grad_mean = None
        if ctx.needs_input_grad[0]:
            backward_factor, = ctx.saved_tensors
            grad_input = grad_output * backward_factor
        return grad_input, grad_factor, grad_mean


class FixedPatternNoiseExponential(torch.autograd.Function):
    """
    Compute exponential function of noisy values. In the backwards-pass, their
    mean value (of the underlying distribution they are sampled from) is
    injected.
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor,
                mean: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mean is not None:
            ctx.save_for_backward(mean)
        else:
            ctx.save_for_backward(input)
        return torch.exp(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_mean = None
        if ctx.needs_input_grad[0]:
            backward_value, = ctx.saved_tensors
            grad_input = grad_output * torch.exp(backward_value)
        return grad_input, grad_mean
