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


def linear_sparse(input: torch.Tensor, weight: torch.nn.parameter.Parameter,
                  connections: torch.Tensor = None,
                  bias: torch.nn.parameter.Parameter = None) -> torch.Tensor:
    """
    Wrap `linear` to allow signature inspection. Disable inactive connections
    in weight tensor.

    :param input: The input to be multiplied with the params tensor `weight`.
    :param weight: The weight parameter tensor. This tensor is expected to be
        dense since pytorch, see issue: 4039.
    :param bias: The bias of the linear operation.
    :param connections: A dense boolean connection mask indicating active
        connections. If None, the weight tensor remains untouched.
    """
    if connections is not None:
        weight.data[~connections] = 0.  # pylint: disable=invalid-unary-operand-type
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
