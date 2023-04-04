"""
Implementing the base module HXModule
"""
from __future__ import annotations
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, Tuple, Type, Optional, Union)
from functools import partial
import inspect
import pylogging as logger

import torch

import pygrenade_vx.network.placed_logical as grenade
from _hxtorch_spiking import DataHandle  # pylint: disable=import-error
from hxtorch.spiking.handle import TensorHandle
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment

log = logger.get("hxtorch.spiking.modules")


class HXModule(torch.nn.Module):
    """
    PyTorch module supplying basic functionality for building SNNs on HX.
    """

    output_type: Type = TensorHandle

    def __init__(self, experiment: Experiment,
                 func: Union[Callable, torch.autograd.Function]) -> None:
        """
        :param experiment: Experiment to append layer to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation.
            TODO: Inform about func args
        """
        super().__init__()

        self._func_is_wrapped = False
        self._changed_since_last_run = True

        self.experiment = experiment
        self.func = func
        self.extra_args: Tuple[Any] = tuple()
        self.extra_kwargs: Dict[str, Any] = {}
        self.size: int = None

        self._output_handle = self.output_type()

        # Grenade descriptor
        self.descriptor: Optional[
            grenade.PopulationDescriptor,
            Union[grenade.ProjectionDescriptor, Tuple[
                grenade.ProjectionDescriptor, ...]]] = None

    @property
    def func(self) -> Callable:
        if not self._func_is_wrapped:
            self._func = self._prepare_func(self._func)
            self._func_is_wrapped = True
        return self._func

    @func.setter
    def func(self, function: Callable) -> None:
        """ Assign a PyTorch-differentiable function to the module.

        :param function: The function describing the modules f"""
        self._func = function
        self._func_is_wrapped = False

    @property
    def changed_since_last_run(self) -> bool:
        """
        Getter for changed_since_last_run.

        :returns: Boolean indicating wether module changed since last run.
        """
        return self._changed_since_last_run

    def reset_changed_since_last_run(self) -> None:
        """
        Reset changed_since_last_run. Sets the corresponding flag to false.
        """
        self._changed_since_last_run = False

    def post_process(self, hw_spikes: Optional[DataHandle],
                     hw_cadc: Optional[DataHandle],
                     hw_madc: Optional[DataHandle]) \
            -> Tuple[Optional[torch.Tensor], ...]:
        """
        This methods needs to be overridden for every derived module that
        demands hardware observables.

        :returns: Hardware data represented as torch.Tensors. Note that
            torch.Tensors are required here to enable gradient flow.
        """
        raise NotImplementedError()

    def _wrap_func(self, function):
        # Signature for wrapping
        signature = inspect.signature(function)

        # Wrap all kwargs except for hw_data
        for key, value in self.extra_kwargs.items():
            if key in signature.parameters and key != "hw_data":
                function = partial(function, **{key: value})

        return function, signature

    # pylint: disable=function-redefined, unused-argument
    def _prepare_func(self, function) -> Callable:
        """
        Strips all args and kwargs excluding `input` and `hw_data` from
        self._func. If self._func does not have an `hw_data` keyword argument
        the prepared function will have it. This unifies the signature of all
        functions used in `exec_forward` to `func(input, hw_data=...)`.
        :param function: The function to be used for building the PyTorch
            graph.
        :returns: Returns the member 'func(input, *args, **kwrags,
            hw_data=...)' stripped down to 'func(input, hw_data=...).
        """
        is_autograd_func = isinstance(
            function, torch.autograd.function.FunctionMeta)

        # In case of HW execution and func is autograd func we override forward
        if is_autograd_func and not self.experiment.mock:
            def func(*inputs, hw_data):
                class LocalAutograd(function):
                    @staticmethod
                    def forward(  # pylint: disable=dangerous-default-value
                            ctx, *data, extra_kwargs=self.extra_kwargs):
                        ctx.extra_kwargs = extra_kwargs
                        ctx.save_for_backward(
                            *data, *hw_data if hw_data is not None else None)
                        return hw_data
                return LocalAutograd.apply(*inputs, *self.extra_args)

            return func

        # In case of SW execution and func is autograd func we use forward
        if is_autograd_func and self.experiment.mock:
            # Make new autograd to not change the original one
            class LocalAutograd(function):
                pass
            LocalAutograd.forward, signature = self._wrap_func(
                LocalAutograd.forward)

            # Wrap HW data on demand
            if "hw_data" in signature.parameters:
                def func(inputs, hw_data=None):
                    # TODO: Is repeatitively calling 'partial' an issue?
                    # We need to wrap keyword argument here in order for apply
                    # to work
                    LocalAutograd.forward = partial(
                        LocalAutograd.forward, hw_data=hw_data)
                    return LocalAutograd.apply(*inputs, *self.extra_args)
            else:
                def func(inputs, hw_data=None):
                    return LocalAutograd.apply(*inputs, *self.extra_args)

            return func

        # In case of HW or SW execution but no autograd func we inject hw data
        # as keyword argument
        local_func, signature = self._wrap_func(function)

        # Wrap HW data on demand
        if "hw_data" in signature.parameters:
            def func(inputs, hw_data=None):
                return local_func(*inputs, *self.extra_args, hw_data=hw_data)
        else:
            def func(inputs, hw_data=None):
                return local_func(*inputs, *self.extra_args)

        return func

    # Allow redefinition of builtin in order to be consistent with PyTorch
    # pylint: disable=redefined-builtin
    def forward(self, *input: Union[Tuple[TensorHandle], TensorHandle]) \
            -> TensorHandle:
        """
        Forward method registering layer operation in given experiment. Input
        and output references will hold corresponding data as soon as
        'hxtorch.run' in executed.

        :param input: Reference to TensorHandle holding data tensors as soon
            as required.

        :returns: Returns a Reference to TensorHandle holding result data
            asociated with this layer after 'hxtorch.run' is executed.
        """
        self.experiment.connect(self, input, self._output_handle)
        return self._output_handle

    # Allow redefinition of builtin in order to be consistent with PyTorch
    # pylint: disable=redefined-builtin
    def exec_forward(self, input: Union[Tuple[TensorHandle], TensorHandle],
                     output: TensorHandle,
                     hw_map: Dict[grenade.PopulationDescriptor,
                                  Tuple[torch.Tensor]]) -> None:
        """
        Inject hardware observables into TensorHandles or execute forward in
        mock-mode.
        """
        # Access HW data
        hw_data = hw_map.get(self.descriptor)
        # Need tuple to allow for multiple input
        if not isinstance(input, tuple):
            input = (input,)
        input = tuple(handle.observable_state for handle in input)
        # Forwards function
        out = self.func(input, hw_data=hw_data)
        # We need to unpack into `Handle.put` otherwise we get Tuple[Tuple]
        # in `put`, however, func should not be limited to return type 'tuple'
        out = (out,) if not isinstance(out, tuple) else out
        output.put(*out)
