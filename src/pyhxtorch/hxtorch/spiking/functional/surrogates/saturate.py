"""
A collection of surrogate functions for the `torch.clamp()` function.

There are Classes deriving from `torch.autograd.Function`, that use
`torch.clamp()` in the forward pass, and only use the surrogate gradient in
the backward pass, but also python functions (having a `_func` suffix in their
names) that implement the surrogate in forward direction.
To enable the use of functools.partial on the `apply` methods of the
`torch.autograd.Function` classes, these are wrapped.
"""
# pylint: disable=redefined-builtin, arguments-differ, abstract-method
import torch


class Clamp(torch.autograd.Function):
    """
    Forward-pass uses `torch.clamp()`, but backward pass uses the identity as a
    surrogate gradient.
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


def clamp(input: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) \
        -> torch.Tensor:
    """Wrapper for Clamp.apply()"""
    return Clamp.apply(input, lower, upper)


def exponential_rolloff_func(
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


class ExponentialRolloff(torch.autograd.Function):
    """
    Forward-pass uses `torch.clamp()`, but backward pass uses the gradient of a
    linear function with exponential roll off towards the saturation bounds.
    See `hxtorch.spiking.functional.surrogate.exponential_rolloff` for greater
    detail on the surrogate.
    """

    # pylint: disable=too-many-arguments
    @staticmethod
    def forward(ctx, input: torch.Tensor, lower: torch.Tensor,
                upper: torch.Tensor, rolloff_margin: float = 0.03,
                rolloff_margin_abs: float = 0.05):

        if torch.any(torch.gt(lower, upper)):
            raise ValueError(
                "Lower bounds may not be larger than upper bounds.")

        abs_margin = torch.where(
            torch.logical_or(torch.isinf(upper), torch.isinf(lower)),
            torch.tensor(rolloff_margin_abs).to(input.device),
            (upper - lower) * rolloff_margin)

        # Save for backward
        ctx.save_for_backward(input, lower, upper, abs_margin)

        return torch.clamp(input, min=lower, max=upper)

    # pylint: disable=too-many-locals
    @staticmethod
    def backward(ctx, grad_output):

        grad_input, grad_lower, grad_upper, grad_rolloff_margin, \
            grad_rolloff_margin_abs = None, None, None, None, None
        if not ctx.needs_input_grad[0]:
            return grad_input, grad_lower, grad_upper, grad_rolloff_margin, \
                grad_rolloff_margin_abs

        grad_input = grad_output.clone()

        (input, lower, upper, abs_margin) = ctx.saved_tensors

        lower = lower.expand_as(input)
        upper = upper.expand_as(input)
        abs_margin = abs_margin.expand_as(input)

        lower_transition_point = lower + abs_margin
        upper_transition_point = upper - abs_margin

        mask_lower = torch.le(input, lower_transition_point)
        mask_upper = torch.ge(input, upper_transition_point)
        mask_equal = torch.eq(lower, upper)

        # Lower rolloff derivative
        if mask_lower.any():
            exp_term = torch.exp(
                (input[mask_lower] - lower_transition_point[mask_lower])
                / abs_margin[mask_lower]
            )
            grad_input[mask_lower] = grad_output[mask_lower] * exp_term

        # Upper rolloff derivative
        if mask_upper.any():
            exp_term = torch.exp(
                (upper_transition_point[mask_upper] - input[mask_upper])
                / abs_margin[mask_upper]
            )
            grad_input[mask_upper] = grad_output[mask_upper] * exp_term

        # Constant, if lower == upper
        grad_input[mask_equal] = 0.0

        return grad_input, grad_lower, grad_upper, grad_rolloff_margin, \
            grad_rolloff_margin_abs


def exponential_rolloff(input: torch.Tensor, lower: torch.Tensor,
                        upper: torch.Tensor, rolloff_margin: float = 0.03,
                        rolloff_margin_abs: float = 0.05) -> torch.Tensor:
    """Wrapper for ExponentialRolloff.apply()"""
    return ExponentialRolloff.apply(
        input, lower, upper, rolloff_margin, rolloff_margin_abs)
