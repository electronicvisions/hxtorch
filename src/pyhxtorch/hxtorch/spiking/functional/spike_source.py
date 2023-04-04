"""
Define different input spike sources
"""
import torch


# Allow redefining builtin for PyTorch consistancy
def input_neuron(input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
    """
    Identity input neuron. This forwards only the input tensor and is only used
    for consistancy in `HXInputNeuron` which can also takes autograd
    functions. This enables gradient flow though input layers.

    :param input: Input spike tensor.

    :retuns: Returns the input spike tensor.
    """
    return input
