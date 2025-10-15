"""
Implementing the base module HXModule
"""
from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Callable,
    Tuple,
    Type,
    Optional,
    Union,
)
import abc
import inspect

import torch

import pygrenade_vx as grenade
from hxtorch.core.modules.hx_module import HXBaseModule
from hxtorch.spiking.handle import TensorHandle
from hxtorch.spiking.observables import HXTorchObservables
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment


class HXTorchFunctionMixin:

    def __init__(self) -> None:
        self._func_is_wrapped = False
        self._func_name = None

    def extra_repr(self) -> str:
        """ Add additional information """
        return f"function={self._func_name}, {super().extra_repr()}"

    # pylint: disable=redefined-builtin, unused-argument
    @abc.abstractmethod
    def forward_func(
        self,
        input: TensorHandle,
        hw_data: Optional[Tuple[torch.Tensor]] = None,
    ) -> TensorHandle:
        pass

    @property
    def func(self) -> Callable:
        if not self._func_is_wrapped:
            self._func = self._prepare_func(self.forward_func)
            self._func_is_wrapped = True
        return self._func

    # pylint: disable=function-redefined, unused-argument
    def _prepare_func(self, function) -> Callable:
        """
        Strips all args and kwargs excluding `input` and `hw_data` from
        self._func. If self._func does not have an `hw_data` keyword argument
        the prepared function will have it. This unifies the signature of all
        functions used in `exec_forward` to `func(input, hw_data=...)`.
        :param function: The function to be used for building the PyTorch
            graph.
        :returns: Returns the member 'func(input, *args, **kwargs,
            hw_data=...)' stripped down to 'func(input, hw_data=...).
        """
        # Infer function name
        try:
            self._func_name = function.__name__ if function is not None \
                else None
        except AttributeError:
            self._func_name = "unknown"

        # In case of HW or SW execution but no autograd func we inject hw data
        # as keyword argument
        signature = inspect.signature(function)

        # Wrap HW data on demand
        if "hw_data" in signature.parameters:
            def func(inputs, hw_data=None):
                return function(*inputs, hw_data=hw_data)
        else:
            def func(inputs, hw_data=None):
                return function(*inputs)

        return func


class HXTorchBaseModule(HXTorchFunctionMixin, torch.nn.Module):

    def post_simulation_processing(self, output) -> None:
        pass

    # pylint: disable=abstract-method
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        HXTorchFunctionMixin.__init__(self)
        self.hw_observables = HXTorchObservables()

    # Allow redefinition of builtin in order to be consistent with PyTorch
    # pylint: disable=redefined-builtin
    def forward(
        self,
        *input: Union[Tuple[TensorHandle], TensorHandle],
    ) -> TensorHandle:
        """
        Forward method registering layer operation in given experiment. Input
        and output references will hold corresponding data as soon as
        'hxtorch.run' in executed.
        :param input: Reference to TensorHandle holding data tensors as soon
            as required.
        :returns: Returns a Reference to TensorHandle holding result data
            associated with this layer after 'hxtorch.run' is executed.
        """
        handle = self.output_type()
        self.experiment.connect(self, input, handle)
        return handle

    # Allow redefinition of builtin in order to be consistent with PyTorch
    # pylint: disable=redefined-builtin
    def exec_forward(
        self,
        input: Union[Tuple[TensorHandle], TensorHandle],
        output: TensorHandle,
    ) -> None:
        """
        Inject hardware observables into TensorHandles or execute forward in
        mock-mode.
        """
        # Need tuple to allow for multiple input
        if not isinstance(input, tuple):
            input = (input,)
        # TODO: post_process could be moved into the func and let user decide
        hw_data = None
        if not self.experiment.mock:
            hw_data = self.post_process(
                self.hw_observables,
                self.experiment.runtime_in_s,
            )
        # Execute function
        returned_handle = self.func(input, hw_data=hw_data)
        output.clone(returned_handle)

    def post_process(
        self,
        hw_data: HXTorchObservables,
        runtime: int,
    ) -> Tuple[Optional[torch.Tensor], ...] | None:
        """
        This methods needs to be overridden for every derived module that
        demands hardware observables and is intended to translated hardware-
        affine datatypes returned by grenade into PyTorch tensors.

        :param hw_data: A ``HardwareObservables`` instance holding the hardware
            data assigned to this module.
        :param runtime: The requested runtime of the experiment on hardware in
            us.
        :param dt: The expected temporal resolution in hxtorch.

        :return: Hardware data represented as torch.Tensors. Note that
            torch.Tensors are required here to enable gradient flow.
        """
        raise NotImplementedError


class HXModule(HXTorchBaseModule, HXBaseModule):
    """
    PyTorch module supplying basic functionality for elements of SNNs that do
    have a representation on hardware
    """
    # pylint: disable=abstract-method
    output_type: Type = TensorHandle

    def __init__(
        self,
        experiment: Experiment,
        chip_coordinate: Optional[
            Tuple[grenade.common.ChipOnConnection,
                  grenade.common.ConnectionOnExecutor]] = None,
    ) -> None:
        """
        :param experiment: Experiment to append layer to.
        :param chip_coordinate: Chip coordinate this module is placed on.
        """
        HXTorchBaseModule.__init__(self)
        HXBaseModule.__init__(self, experiment, chip_coordinate)

    def extra_repr(self) -> str:
        """ Add additional information """
        # TODO: move this to pytorch stuff
        reprs = f"experiment={self.experiment}, "
        reprs += f"{super().extra_repr()}"
        return reprs


class HXFunctionalModule(HXTorchBaseModule, HXBaseModule):
    """
    PyTorch module supplying basic functionality for elements of SNNs that do
    not have a direct hardware representation
    """
    # pylint: disable=abstract-method
    output_type: Type = TensorHandle

    def __init__(self, experiment: Experiment) -> None:
        """
        :param experiment: Experiment to append layer to.
        """
        HXBaseModule.__init__(self, experiment)
        HXTorchBaseModule.__init__(self)
