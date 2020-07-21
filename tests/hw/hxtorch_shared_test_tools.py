"""
Some code shared between tests.
"""
import torch


def rand_full(size, mean_value) -> torch.Tensor:
    """
    Returns a test tensor that requires a gradient.
    It will have the given size and uniform distributed entries ±20% of the
    provided mean value.
    This can be used to test the gradient for different mean values while
    tracking each individual entry.
    """
    return torch.empty(size, dtype=torch.float).uniform_(
        .8 * mean_value, 1.2 * mean_value).requires_grad_()
