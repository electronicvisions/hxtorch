"""
Implement linear (and similar) autograd functions
"""
import torch
from dlens_vx_v3 import lola


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

    :param input: The input neuron tensor holding spikes to be multiplied with
        the params tensor `weight`.
    :param weight: The weight parameter tensor. This tensor is expected to be
        dense since pytorch, see issue: 4039.
    :param bias: The bias of the linear operation.
    :param connections: A dense boolean connection mask indicating active
        connections. If None, the weight tensor remains untouched.
    """
    if connections is not None:
        weight.data[~connections] = 0.  # pylint: disable=invalid-unary-operand-type
    return torch.nn.functional.linear(input, weight, bias)


# pylint: disable=too-many-arguments
def linear_exponential_clamp(inputs: torch.Tensor,
                             weight: torch.nn.parameter.Parameter,
                             bias: torch.nn.parameter.Parameter = None,
                             cap: float = 1.5, start_weight: float = 61.,
                             quantize: bool = False) -> torch.Tensor:
    """
    Clamps the weights with an exponential roll-off towards saturation.

    :param input: The input neuron tensor holding spikes to be multiplied with
        the params tensor `weight`.
    :param weight: Weight Tensor to be clamped.
    :param bias: The bias of the linear operation.
    :param cap: Upper resp. -1 * lower boundary of the weights. Choose this
        value to be 1 / weight_scaling (see hxtorch.spiking.Synapse) to
        saturate the software weights where theirs scaled values saturate on
        hardware.
    :param start_weight: Indicating at which hardware-weight the roll off
        begins. Has to be in range (0, 63).
    :param quantize: If true, the weights are rounded to multiples of cap / 63
        to match the discrete hardware representation.

    :return: Clamped weights and possibly rounded weights
    """
    max_weight = lola.SynapseWeightMatrix.Value.max
    steepness = 1 / (1 - start_weight / max_weight)

    clamped_weight = torch.where(
        torch.abs(weight) > start_weight / max_weight * cap,
        torch.sign(weight) * cap * (1 - 1 / steepness * torch.exp(
            - steepness * (torch.abs(weight) / cap - (1 - 1 / steepness)))),
        weight)

    if quantize:
        clamped_weight = WeightRounding.apply(cap, clamped_weight)

    return torch.nn.functional.linear(inputs, clamped_weight, bias)


# Disable abstract-method because pylint does not check meta classes
# pylint: disable=abstract-method, arguments-differ, unused-argument
class WeightRounding(torch.autograd.Function):
    """
    A function that rounds the input to multiples of cap / 63 but uses a linear
    function (the identity) in the backward path
    """

    @staticmethod
    def forward(ctx, cap, weight):
        max_weight = lola.SynapseWeightMatrix.Value.max
        disc_w = torch.round(weight / (cap / max_weight)) * cap / max_weight
        return disc_w

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output
