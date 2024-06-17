"""
Implementing the base module HXModule
"""
from __future__ import annotations
import abc
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, Tuple, Type, Optional, Union)
from functools import partial
import inspect
import pylogging as logger

import torch

from dlens_vx_v3 import lola, halco
import pygrenade_vx.network as grenade
from hxtorch.spiking.handle import TensorHandle, NeuronHandle
if TYPE_CHECKING:
    from hxtorch.spiking.execution_instance import ExecutionInstance
    from hxtorch.spiking.observables import HardwareObservables
    from hxtorch.spiking.experiment import Experiment
    from hxtorch.spiking.observables import HardwareObservables

log = logger.get("hxtorch.spiking.modules")


class HXBaseExperimentModule(torch.nn.Module):

    output_type: Type = TensorHandle
    descriptor = None

    def __init__(self, experiment: Experiment) -> None:
        """
        :param experiment: Experiment to append layer to.
        """
        super().__init__()
        self.experiment = experiment
        self._output_handle = self.output_type()

    def extra_repr(self) -> str:
        """ Add additional information """
        return f"experiment={self.experiment}"

    @property
    @abc.abstractmethod
    def func(self) -> Callable:
        """
        Getter for function assigned to the module

        :return: Returns the function assigned to the module.
        """

    @func.setter
    @abc.abstractmethod
    def func(self, function: Callable) -> None:
        """
        Assign a PyTorch-differentiable function to the module. This function
        is used in mock-mode or gets the hardware observables injected in non-
        mock-mode if the function provides a keyword argument 'hw_data'.

        :param function: The function describing the modules
        """

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
            associated with this layer after 'hxtorch.run' is executed.
        """
        self.experiment.connect(self, input, self._output_handle)
        return self._output_handle

    # Allow redefinition of builtin in order to be consistent with PyTorch
    # pylint: disable=redefined-builtin
    def exec_forward(self, input: Union[Tuple[TensorHandle], TensorHandle],
                     output: TensorHandle,
                     hw_map: Dict[
                         grenade.network.PopulationsDescriptor, Any]) -> None:
        """
        Inject hardware observables into TensorHandles or execute forward in
        mock-mode.
        """
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


class HXTorchFunctionMixin:

    def __init__(self, func: Union[Callable, torch.autograd.Function]) -> None:
        """
        :param func: Callable function implementing the module's forward
            functionality or a torch.autograd.Function implementing the
            module's forward and backward operation.
            TODO: Inform about func args
        """
        self._func_is_wrapped = False
        self._func_name = None

        self.func = func
        self.extra_args: Tuple[Any] = tuple()
        self.extra_kwargs: Dict[str, Any] = {}

    def extra_repr(self) -> str:
        """ Add additional information """
        return f"function={self._func_name}, {super().extra_repr()}"

    @property
    def func(self) -> Callable:
        if not self._func_is_wrapped:
            self._func = self._prepare_func(self._func)
            self._func_is_wrapped = True
        return self._func

    @func.setter
    def func(self, function: Optional[Callable]) -> None:
        """
        Assign a PyTorch-differentiable function to the module. This function
        is used in mock-mode or gets the hardware observables injected in non-
        mock-mode if the function provides a keyword argument 'hw_data'.

        :param function: The function describing the modules
        """
        self._func = function
        try:
            self._func_name = function.__name__ if function is not None \
                else None
        except AttributeError:
            self._func_name = "unknown"
        self._func_is_wrapped = False

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
        :returns: Returns the member 'func(input, *args, **kwargs,
            hw_data=...)' stripped down to 'func(input, hw_data=...).
        """
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


class HXHardwareEntityMixin:

    def __init__(self, execution_instance: Optional[ExecutionInstance] = None)\
            -> None:
        """
        :param execution_instance: Execution instance to place to.
        """
        self._changed_since_last_run = True

        if execution_instance is None and not self.experiment.mock:
            execution_instance = self.experiment.default_execution_instance
        self.execution_instance = execution_instance

        # Grenade descriptor
        self.descriptor: Optional[Union[
            grenade.PopulationOnNetwork, grenade.ProjectionOnNetwork,
            Tuple[grenade.ProjectionOnNetwork, ...]]] = None

    def extra_repr(self) -> str:
        """ Add additional information """
        reprs = f"execution_instance={self.execution_instance}, "
        reprs += f"{super().extra_repr()}"
        return reprs

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

    def add_to_input_generator(  # pylint: disable=redefined-builtin
            self, input: NeuronHandle,
            builder: grenade.network.InputGenerator) -> None:
        """
        Add the input to an input module to grenades input generator.

        :param module: The module to add the input for.
        :param builder: Grenade's logical network builder.
        """

    def post_process(self, hw_data: HardwareObservables, runtime: float) \
            -> Tuple[Optional[torch.Tensor], ...]:
        """
        This methods needs to be overridden for every derived module that
        demands hardware observables and is intended to translated hardware-
        affine datatypes returned by grenade into PyTorch tensors.

        :param hw_data: A ``HardwareObservables`` instance holding the hardware
            data assigned to this module.
        :param runtime: The requested runtime of the experiment on hardware in
            s.
        :param dt: The expected temporal resolution in hxtorch.

        :return: Hardware data represented as torch.Tensors. Note that
            torch.Tensors are required here to enable gradient flow.
        """

    def register_hw_entity(self) -> None:
        """ Register Module in member `Experiment` """

    def configure_hw_entity(self, neuron_id: int,
                            neuron_block: lola.NeuronBlock,
                            coord: halco.LogicalNeuronOnDLS) \
            -> lola.NeuronBlock:
        """
        Configures a neuron in the given layer with its specific properties.
        The neurons digital event outputs are enabled according to the given
        spiking mask.

        :param neuron_id: In-population neuron index.
        :param neuron_block: The neuron block hardware entity.
        :param coord: Coordinate of neuron on hardware.
        :returns: Configured neuron block.
        """


class HXModule(
        HXTorchFunctionMixin, HXHardwareEntityMixin, HXBaseExperimentModule):
    """
    PyTorch module supplying basic functionality for elements of SNNs that do
    have a representation on hardware
    """

    def __init__(self, experiment: Experiment,
                 func: Union[Callable, torch.autograd.Function],
                 execution_instance: Optional[ExecutionInstance] = None) \
            -> None:
        """
        :param experiment: Experiment to append layer to.
        :param func: Callable function implementing the module's forward
            functionality or a torch.autograd.Function implementing the
            module's forward and backward operation.
            TODO: Inform about func args
        :param execution_instance: Execution instance to place to.
        """
        HXBaseExperimentModule.__init__(self, experiment)
        HXTorchFunctionMixin.__init__(self, func)
        HXHardwareEntityMixin.__init__(self, execution_instance)


class HXFunctionalModule(HXTorchFunctionMixin, HXBaseExperimentModule):
    """
    PyTorch module supplying basic functionality for elements of SNNs that do
    not have a direct hardware representation
    """

    def __init__(self, experiment: Experiment,
                 func: Union[Callable, torch.autograd.Function]) -> None:
        """
        :param experiment: Experiment to append layer to.
        :param func: Callable function implementing the module's forward
            functionality or a torch.autograd.Function implementing the
            module's forward and backward operation.
            TODO: Inform about func args
        """
        HXBaseExperimentModule.__init__(self, experiment)
        HXTorchFunctionMixin.__init__(self, func)
