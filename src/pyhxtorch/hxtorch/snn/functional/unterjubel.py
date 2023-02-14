"""
Autograd function to 'unterjubel' (german for 'inject') hardware observables
and allow correct gradient backpropagation.

Basically taken from:
https://gerrit.bioai.eu:9443/gitweb?p=model-hx-strobe.git;a=blob;f=src/py/strobe/unterjubel.py;h=4b13de159bd54b629a0b6278cb9c2f58f62883bf;hb=HEAD;js=1
"""
from typing import Tuple, Optional
import torch


# Issue: 4019
# pylint: disable=abstract-method
class Unterjubel(torch.autograd.Function):

    """ Unterjubel hardware observables to allow correct gradient flow """

    # Allow redefining builtin for PyTorch consistancy
    # pylint: disable=redefined-builtin, arguments-differ, unused-argument
    @staticmethod
    def forward(ctx, input: torch.Tensor, input_prime: torch.Tensor) \
            -> torch.Tensor:
        """
        Returns `input_prime` insteat of `input` to inject `input_prime` but
        direct the gradient to `input`.

        :param input: Input tensor.
        :param input_prime: The returned tensor.

        :returns: Returns the primed tensor. Thereby, this tensor is forwarded
            while the gradient is directed to to `input`.
        """
        return input_prime

    # Allow redefining builtin for PyTorch consistancy
    # pylint: disable=redefined-builtin, arguments-differ
    @staticmethod
    def backward(ctx: torch.Tensor, grad_output: torch.Tensor) \
            -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward the gradient.

        :param grad_output: The backwarded gradient.

        :returns: Returns simply the backpropagated gradient at first position.
        """
        return grad_output, None, None
