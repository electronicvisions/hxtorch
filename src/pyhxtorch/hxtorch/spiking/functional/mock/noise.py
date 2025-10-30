"""
Define dataclass and torch.autograd.Function to mock random noise
"""
# pylint: disable=redefined-builtin, arguments-differ, abstract-method
from __future__ import annotations
from typing import Optional, Tuple, Union, Self
from dataclasses import dataclass
import torch


@dataclass
class RandomNoise:
    """
    Class defining gaussian random noise to mock the random noise of the
    membrane and adaptation state on hardware along the time axis.
    It is entirely defined by the standard deviation of a gaussian centered
    around 0, which can be sampled from.
    """
    std: Union[float, torch.Tensor, None] = None
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        """
        Transform floats to torch.Tensor (since they are passed to
        torch.autograd.Function)
        """
        if self.std is not None:
            self.std = torch.as_tensor(self.std)

    def __getitem__(self, idx) -> RandomNoise:
        if isinstance(self.std, torch.Tensor):
            return RandomNoise(std=self.std[idx], device=self.device)
        raise ValueError(
            f"RandomNoise object with id {id(self)} is not subscriptable.")

    def sample(self, size: Union[Tuple[int], int, None] = None) \
            -> torch.Tensor:
        """
        Sample random noise.
        :param size: The size of the sampled tensor. If None is passed, the
            resulting tensor is of the same size as `self.std`
        :returns: Returns a tensor of random values drawn from a gaussian with
            standard deviation `self.std` and mean zero.
        """
        if self.std is None:
            random_tensor = torch.zeros(size) if size else torch.zeros(1)
        else:
            if size is None:
                mean = torch.zeros_like(self.std)
                random_tensor = torch.normal(mean=mean, std=self.std)
            else:
                mean = torch.zeros(size)
                std = self.std.expand(size)
                random_tensor = torch.normal(mean=mean, std=std)
        if self.device is None:
            return random_tensor
        return random_tensor.to(self.device)

    # pylint: disable=invalid-name
    def to(self, device: torch.device) -> Self:
        """
        Set the device, the sampled tensors are going to be assigned to.
        :param device: The device, which is to be set.
        """
        self.device = device
        return self


class RandomNoiseAdd(torch.autograd.Function):
    """ Add random noise to a tensor """

    # pylint: disable=unused-argument
    @staticmethod
    def forward(
            ctx, input: torch.Tensor, noise: Optional[torch.Tensor]) \
            -> torch.Tensor:
        if noise is None:
            return input
        return input + noise

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_noise = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        return grad_input, grad_noise
