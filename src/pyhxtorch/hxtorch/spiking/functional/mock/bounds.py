"""
Define dataclass that holds data needed to mock bounds (e.g. bounds of the
dynamic ranges of membrane- and adaptation voltage or the CADC readout bounds)
"""
from __future__ import annotations
from typing import Optional, Union, Self, Callable
from dataclasses import InitVar, dataclass, field
from functools import partial
import torch

from hxtorch.spiking.functional.surrogates import exponential_rolloff


@dataclass
class Bounds:
    """
    Class defining bounds of a finite value range and how to deal with the
    bounds in simulation.
    :param lower: The lower bound.
    :param upper: The upper bound.
    :param device: The device, the tensors containing the bounds are
        transfered to.
    :param hardware_aware: When set to True, the saturation effect is
        considered in the backward pass of the simulation; Else, the backward
        function is set to be the identity function and `torch.clamp()` is used
        instead of a surrogate.
    :param surrogate: Callable that implements a surrogate function for the
        clamp function. Is needed in case of of a hardware aware
        backpropagation.
    """
    lower: InitVar[Union[float, torch.Tensor]]
    _lower: torch.Tensor = field(init=False)
    upper: InitVar[Union[float, torch.Tensor]]
    _upper: torch.Tensor = field(init=False)
    device: Optional[torch.device] = None
    hardware_aware: bool = True
    surrogate: Callable = partial(
        exponential_rolloff, rolloff_margin=0.1, rolloff_margin_abs=0.15)

    def __post_init__(self, lower: Union[float, torch.Tensor],
                      upper: Union[float, torch.Tensor]) -> None:
        """
        Transform floats to torch.Tensor (since they are passed to
        torch.autograd.Function)
        """
        self._lower = torch.as_tensor(lower)
        self._upper = torch.as_tensor(upper)
        if self.device is not None:
            self.to(self.device)

    def __getitem__(self, idx) -> Bounds:
        return Bounds(
            lower=self._lower[idx], upper=self._upper[idx], device=self.device,
            hardware_aware=self.hardware_aware, surrogate=self.surrogate)

    @property
    def lower(self) -> torch.Tensor:
        return self._lower

    @lower.setter
    def lower(self, lower: Union[float, torch.Tensor]) -> None:
        self._lower = torch.as_tensor(lower)

    @property
    def upper(self) -> torch.Tensor:
        return self._upper

    @upper.setter
    def upper(self, upper: Union[float, torch.Tensor]) -> None:
        self._upper = torch.as_tensor(upper)

    # pylint: disable=invalid-name
    def to(self, device: torch.device) -> Self:
        """
        Set the device of the tensors containing the bound values
        :param device: The device to transfer the tensors containing the
            bound values to.
        """
        self.device = device
        self._lower = self._lower.to(device)
        self._upper = self._upper.to(device)
        return self
