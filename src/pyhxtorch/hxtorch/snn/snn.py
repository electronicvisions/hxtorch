"""
Implementing SNN modules
"""
# pylint: disable=no-member
from typing import Any, Callable, Dict, Tuple, Type, Optional, Union
from functools import partial
import inspect
import numpy as np

import torch
import torch.nn.functional as nnF
from torch.nn.parameter import Parameter

import hxtorch.snn.functional as F
from hxtorch.snn.handle import (
    TensorHandle, NeuronHandle, ReadoutNeuronHandle, SynapseHandle)
from hxtorch.snn.instance import Instance


class HXModule(torch.nn.Module):
    """
    PyTorch module supplying basic functionality for building SNNs on HX.
    """

    output_type: Type = TensorHandle

    def __init__(self, instance: Instance,
                 func: Union[Callable, torch.autograd.Function]) -> None:
        """
        :param instance: Instance to append layer to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation.
            TODO: Inform about func args
        """
        super().__init__()

        self._func = func
        self._instance = instance
        self.extra_args: Optional[Tuple[Any]] = ()
        self.extra_kwargs: Optional[Dict[str, Any]] = {}
        self.params = None
        self.size: int = None

    def _is_autograd_fn(self) -> bool:
        """
        Determines whether the used function `func` is an autograd function or
        not.

        :returns: Returns whether `func` is an autograd function.
        """
        return isinstance(self._func, torch.autograd.function.FunctionMeta)

    def prepare_func(self, hw_data: torch.Tensor) -> Callable:
        """
        Strips `param` and `hw_result` arguments from self._func. This allows
        having a more general signature of self._func in `exec_forward`.

        :param hw_result: HW observables returned from grenade.

        :returns: Returns the memeber 'func(..., params=..., hw_result=...)'
            stripped down to 'func(...).
        """
        self.extra_kwargs.update({"hw_data": hw_data})
        if self._is_autograd_fn():
            class LocalAutograd(self._func):
                pass
            local_func = LocalAutograd
            signature = inspect.signature(local_func.forward)
            for key, value in self.extra_kwargs.items():
                if key in signature.parameters:
                    local_func.forward = partial(
                        local_func.forward, **{key: value})
            return local_func.apply

        signature = inspect.signature(self._func)
        local_func = self._func
        for key, value in self.extra_kwargs.items():
            if key in signature.parameters:
                local_func = partial(local_func, **{key: value})
        return local_func

    # Allow redefinition of builtin in order to be consistent with PyTorch
    # pylint: disable=redefined-builtin
    def forward(self, *input: Union[Tuple[TensorHandle], TensorHandle]) \
            -> TensorHandle:
        """
        Forward method registering layer operation in given instance. Input and
        output references will hold corresponding data as soon as 'hxtorch.run'
        in executed.

        :param input: Reference to TensorHandle holding data tensors as soon
            as required.

        :returns: Returns a Reference to TensorHandle holding result data
            asociated with this layer after 'hxtorch.run' is executed.
        """
        output = self.output_type()
        self._instance.connect(self, input, output)

        return output

    # Allow redefinition of builtin in order to be consistent with PyTorch
    # pylint: disable=redefined-builtin
    def exec_forward(self, input: Union[Tuple[TensorHandle], TensorHandle],
                     output: TensorHandle,
                     hw_data: Optional[Tuple[torch.Tensor]] = None) -> None:
        """
        Inject hardware observables into TensorHandles or execute forward in
        mock-mode.
        """
        # Need tuple to allow for multiple input
        if not isinstance(input, tuple):
            input = (input,)
        input = tuple(handle.observable_state for handle in input)

        if self._is_autograd_fn() and not self._instance.mock:

            class LocalAutograd(self._func):
                @staticmethod
                def forward(  # pylint: disable=dangerous-default-value
                        ctx, *data, extra_kwargs=self.extra_kwargs):
                    ctx.hw_data = hw_data
                    ctx.extra_kwargs = extra_kwargs
                    ctx.save_for_backward(data)
                    return hw_data

            out = LocalAutograd.apply(*(input + self.extra_args))
        else:
            out = self.prepare_func(hw_data)(*input, *self.extra_args)

        out = (out,) if not isinstance(out, tuple) else out
        output.put(*out)


class Synapse(HXModule):
    """
    A Synapse layer

    Caveat:
    For execution on hardware, this module can only be used in conjuction with
    a subsequent Neuron module.
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
    output_type: Type = SynapseHandle

    # pylint: disable=too-many-arguments
    def __init__(self, in_features: int, out_features: int, instance: Instance,
                 func: Union[Callable, torch.autograd.Function] = nnF.linear,
                 device: str = None, dtype: Type = None) \
            -> None:
        """
        :param in_features: Size of input dimension.
        :param out_features: Size of output dimension.
        :param device: Device to execute on. Only considered in mock-mode.
        :param dtype: Data type of weight tensor.
        :param instance: Instance to append layer to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation. Required function args:
                [input (torch.Tensor), weight (torch.Tensor)]
        """
        super().__init__(instance=instance, func=func)

        self.in_features = in_features
        self.out_features = out_features
        self.size = out_features

        self.weight = Parameter(
            torch.empty((out_features, in_features), device=device,
                        dtype=dtype))

        self.reset_parameters(1.0e-3, 1. / np.sqrt(in_features))
        self.extra_args = (self.weight, None)

    def reset_parameters(self, mean: float, std: float) -> None:
        """
        Resets the modules parameters. Parameters are sampled from a normal
        distribution with mean `mean` and standard deviation `std`.

        :param mean: Mean of normal distribution.
        :param std: Standard deviation of normal distribution.
        """
        torch.nn.init.normal_(self.weight, mean=mean, std=std)


class Neuron(HXModule):
    """
    Neuron layer

    Caveat:
    For execution on hardware, this module can only be used in conjuction with
    a preceding Synapse module.
    """

    output_type: Type = NeuronHandle

    # pylint: disable=too-many-arguments
    def __init__(self, size: int, instance: Instance,
                 func: Union[Callable, torch.autograd.Function] = F.LIF,
                 params: Optional[Union[F.LIFParams, F.LIParams]] = None) \
            -> None:
        """
        Initialize Neuron

        :param instance: Instance to append layer to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation.
        :param params: Neuron Parameters in case of mock neuron integration of
            for backward path. If func does have a param argument the params
            object will get injected automatically.
        """
        super().__init__(instance=instance, func=func)

        self.size = size
        self.params = params
        self.extra_kwargs.update({"params": params})

    # pylint: disable=redefined-builtin, arguments-differ, useless-super-delegation
    # pylint thinks this is useless, but we override signature for
    # documentation purposes (Neurons gets a single SynapseHandle as input)
    def forward(self, input: SynapseHandle) -> NeuronHandle:
        """
        TODO: Remove. Just temporary to infere size impliciltly.
        """
        return super().forward(input)


class ReadoutNeuron(Neuron):
    """
    Readout neuron layer

    Caveat:
    For execution on hardware, this module can only be used in conjuction with
    a preceding Synapse module.
    """
    output_type: Type = ReadoutNeuronHandle


class Dropout(HXModule):
    """
    Batch dropout layer.

    Caveat:
    In-place operations on TensorHandles are not supported. Must be placed
    after a neuron layer, i.e. Neuron.
    """

    output_type: Type = NeuronHandle

    def __init__(self, size: int, dropout: float, instance: Instance,
                 func: Union[Callable, torch.autograd.Function] = nnF.dropout)\
            -> None:
        """
        Initialize Dropout layer.

        :param size: Size of the population this dropout layer is applied to.
        :param dropout: Probability that a neuron in the precessing layer gets
            disabled during training.
        :param instance: Instance to append layer to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation.
        :param params: Neuron Parameters in case of mock neuron integration of
            for backward path. If func does have a param argument the params
            object will get injected automatically.
        """
        super().__init__(instance=instance, func=func)

        self.size = size
        self.extra_args = (dropout, self.training, False)

    # pylint: disable=redefined-builtin, arguments-differ, useless-super-delegation
    # pylint thinks this is useless, but we override signature for
    # documentation purposes (Neurons gets a single SynapseHandle as input)
    def forward(self, input: NeuronHandle) -> NeuronHandle:
        """
        TODO: Remove. Just temporary to infere size impliciltly.
        """
        return super().forward(input)
