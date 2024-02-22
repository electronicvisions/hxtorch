"""
Define different input spike sources
"""
import torch


# Allow redefining builtin for PyTorch consistency
def input_neuron(input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
    """
    Identity input neuron. This forwards only the input tensor and is only used
    for consistency in `HXInputNeuron` which can also takes autograd
    functions. This enables gradient flow though input layers.

    :param input: Input spike tensor.

    :returns: Returns the input spike tensor.
    """
    return input
