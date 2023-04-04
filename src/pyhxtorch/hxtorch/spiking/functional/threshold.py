"""
Threshold function providing functionality for making spiking outputs
differentiable.
Inspired by https://github.com/norse/norse/blob/
    72a812e17da23487878a667ad82a075ef7ad91ec/norse/torch/functional/
    superspike.py
"""
import torch
from hxtorch.spiking.functional.superspike import superspike_func


# Allow redefining builtin for PyTorch consistancy
# pylint: disable=redefined-builtin, invalid-name, too-many-locals
def threshold(input: torch.Tensor, method: str, alpha: float) -> torch.Tensor:
    """
    Selection of the used threshold function.
    :param input: Input tensor to threshold function.
    :param method: The string indicator of the the threshold function.
        Currently supported: 'super_spike'.
    :param alpha: Parameter controlling the slope of the surrogate
        derivative in case of 'superspike'.
    :return: Returns the tensor of the threshold function.
    """
    if method == "superspike":
        return superspike_func(input, torch.as_tensor(alpha))
    raise ValueError(
        f"Threshold function '{method}' is not implemented yet")
