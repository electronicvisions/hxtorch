"""
Define convenient software <-> hardware weight mappings

TODO: Implement stochastic rounding?
"""
import torch


def linear_saturating(weight: torch.Tensor, scale: float = 1.,
                      min_weight: float = -63., max_weight: float = 63.) \
        -> torch.Tensor:
    """
    Scale all weights according to:

        w <- clip(scale * w, min_weight, max_weight)

    TODO: Maybe make this function member of HXSynapse and allow different
          weight transformations by inheritance.

    :param weight: The weight tensor to be transformed.
    :param scale: A constant the weight tensor is scaled with.
    :param min_weight: The minimum value, smaller values are clipped to after
        scaling.
    :param max_weight: The maximum value, bigger values are clipped to after
        scaling.

    :returns: The transformed weight tensor.
    """
    return torch.clamp(scale * weight, min_weight, max_weight)


def linear_saturation_and_rounding(weight: torch.Tensor, scale: float = 1.,
                                   min_weight: float = -63.,
                                   max_weight: float = 63.) \
        -> torch.Tensor:
    """
    Scale all weights according to:

        w <- clip(scale * w, min_weight, max_weight)

    and then round to the nearest integer.

    :param weight: The weight tensor to be transformed.
    :param scale: A constant the weight tensor is scaled with.
    :param min_weight: The minimum value, smaller values are clipped to after
        scaling.
    :param max_weight: The maximum value, bigger values are clipped to after
        scaling.

    :returns: The transformed weight tensor.
    """
    return torch.round(torch.clamp(scale * weight, min_weight, max_weight))


def stochastic_rounding_and_saturation(weight: torch.Tensor,
                                       scale: float = 1.,
                                       min_weight: float = -63.,
                                       max_weight: float = 63.) \
        -> torch.Tensor:
    """
    Scale all weights according to:

        w <- clip(scale * w, min_weight, max_weight)

    and then apply stochastic rounding, i.e. having a float x, round up with
    probability p_up and round down with p_down, like

        p_up = x - floor(x)
        p_down = 1 - p_up = ceil(x) - x

    :param weight: The weight tensor to be transformed.
    :param scale: A constant the weight tensor is scaled with.
    :param min_weight: The minimum value, smaller values are clipped to after
        scaling.
    :param max_weight: The maximum value, bigger values are clipped to after
        scaling.

    :returns: The transformed weight tensor.
    """
    with torch.no_grad():
        clamped_weight = torch.clamp(scale * weight, min_weight, max_weight)

        p_up = clamped_weight - torch.floor(clamped_weight)

        probs = torch.stack(((1. - p_up).flatten(), p_up.flatten())).T

        rounded_weight = torch.floor(weight) + \
            torch.multinomial(probs, num_samples=1,
                              replacement=True).reshape(p_up.shape)

    return rounded_weight
