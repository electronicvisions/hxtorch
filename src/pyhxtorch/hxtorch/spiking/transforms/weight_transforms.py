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
