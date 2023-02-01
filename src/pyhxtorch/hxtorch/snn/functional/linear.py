"""
Implement linear autograd function
"""
import torch


# Allow redefining builtin for PyTorch consistancy
# pylint: disable=redefined-builtin, invalid-name
def linear(input: torch.Tensor, weight: torch.nn.parameter.Parameter,
           bias: torch.nn.parameter.Parameter = None) -> torch.Tensor:
    """
    Wrap `linear` to allow signature inspection
    """
    return torch.nn.functional.linear(input, weight, bias)


class Linear(torch.autograd.Function):

    """ Linear autograd example """

    # Allow redefining builtin for PyTorch consistancy
    # pylint: disable=arguments-differ, disable=redefined-builtin
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor):
        """ Gets overriden """

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Implement linear backward

        :param grad_output: The backpropagted gradient tensor.
        """
        raise NotImplementedError()
