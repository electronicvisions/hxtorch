"""
Define a saturation function needed to mock bounds (e.g. bounds of the dynamic
ranges of membrane- and adaptation voltage or the CADC readout bounds)
"""
# pylint: disable=redefined-builtin
import torch

from hxtorch.spiking.functional.mock.bounds import Bounds
from hxtorch.spiking.functional.surrogates import clamp


def saturate(input: torch.Tensor, bounds: Bounds) -> torch.Tensor:
    """
    Applies thresholds.
    :param input: Tensor, to which the thresholds are applied to.
    :param bounds: Bounds object containing thesholds and the function, which
        is used to apply the thresholds.
    """
    if not bounds.hardware_aware:
        return clamp(input, bounds.lower, bounds.upper)
    return bounds.surrogate(input, bounds.lower, bounds.upper)
