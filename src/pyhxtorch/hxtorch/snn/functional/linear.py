"""
Implement linear autograd function
"""
import torch


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
