"""
Define a saturation function needed to mock bounds (e.g. bounds of the dynamic
ranges of membrane- and adaptation voltage or the CADC readout bounds)
"""
# pylint: disable=redefined-builtin, arguments-differ, abstract-method
import torch

from hxtorch.spiking.functional.mock.bounds import Bounds


def saturate(input: torch.Tensor, bounds: Bounds) -> torch.Tensor:
    """
    Applies soft or hard thresholds.
    :param input: Tensor, to which the thresholds are applied to.
    :param bounds: Bounds object containing thesholds and information about how
        the thresholds are to be applied to the input.
    """
    if not bounds.hardware_aware:
        return Clamp.apply(input, bounds.lower, bounds.upper)
    match bounds.surrogate:
        case "linear_rolloff":
            return linear_rolloff(
                input, bounds.lower, bounds.upper, bounds.rolloff_margin,
                bounds.rolloff_margin_abs)
        case _:
            raise ValueError(
                f"No matching function for surrogate '{bounds.surrogate}'.")


class Clamp(torch.autograd.Function):
    """
    Clamp values to given bounds, if exceeded.
    """

    # pylint: disable=unused-argument
    @staticmethod
    def forward(ctx, input: torch.Tensor, lower: torch.Tensor,
                upper: torch.Tensor) -> torch.Tensor:
        """
        Clamps values to thresholds.
        :param input: Tensor, to which the thresholds are applied to.
        :param lower: Lower threshold.
        :param upper: Upper threshold.
        """
        return torch.clamp(input, min=lower, max=upper)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_lower = grad_upper = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        return grad_input, grad_lower, grad_upper


def linear_rolloff(
        input: torch.Tensor, lower: torch.Tensor,
        upper: torch.Tensor, rolloff_margin: float = 0.03,
        rolloff_margin_abs: float = 0.05) -> torch.Tensor:
    """
    Linear function capped at lower- and upper bounds with a roll off between
    the linear and the constant sections.
    :param input: Tensor, to which the function is applied to.
    :param lower: Lower threshold. If infinite, the roll off on the lower end
        is not applied at all.
    :param upper: Upper threshold.
    :param rolloff_margin: Size of the margin from the bounds inwards, in which
        the roll off is active. Value relative to the distance between the
        bounds.
    :param rolloff_margin_abs: Absolute size of the margin from the bounds
        inwards, in which the roll off is active. This value is needed as a
        fallback, in case one of the thresholds is infinite.
    """
    # Prevent lots of undefined behaviour by forbidding senseless bounds
    if torch.any(torch.gt(lower, upper)):
        raise ValueError("Lower bounds may not be larger than upper bounds.")
    lower = torch.broadcast_to(lower, input.shape)
    upper = torch.broadcast_to(upper, input.shape)
    rolloff_margin_abs = torch.full(input.shape, rolloff_margin_abs)
    abs_margin = torch.where(
        torch.logical_or(upper.isinf(), lower.isinf()), rolloff_margin_abs,
        (upper - lower) * rolloff_margin)
    lower_transition_point = lower + abs_margin
    upper_transition_point = upper - abs_margin
    rolloff_lower = torch.zeros_like(input)
    rolloff_lower[~lower.isinf()] = lower[~lower.isinf()] \
        + abs_margin[~lower.isinf()] * torch.exp(
            (input[~lower.isinf()] - lower_transition_point[~lower.isinf()])
            / abs_margin[~lower.isinf()])
    rolloff_upper = torch.zeros_like(input)
    rolloff_upper[~upper.isinf()] = upper[~upper.isinf()] \
        - abs_margin[~upper.isinf()] * torch.exp(
            (upper_transition_point[~upper.isinf()] - input[~upper.isinf()])
            / abs_margin[~upper.isinf()])
    output = input
    output = torch.where(torch.ge(output, lower_transition_point), output,
                         rolloff_lower)
    output = torch.where(torch.le(output, upper_transition_point), output,
                         rolloff_upper)
    # Treat special case, where lower is equal to upper
    output = torch.where(torch.eq(lower, upper), lower, output)
    return output
