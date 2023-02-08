"""
Implementing a module wrapper to wrap multiple modules as one
"""
# pylint: disable=too-many-lines
from typing import Callable, Dict, Tuple, Optional, List

import torch
import pygrenade_vx as grenade

import hxtorch
from hxtorch.snn.handle import TensorHandle
from hxtorch.snn.modules.hx_module import HXModule

log = hxtorch.logger.get("hxtorch.snn.modules")


class HXModuleWrapper(HXModule):  # pylint: disable=abstract-method
    """ Class to wrap HXModules """

    def __init__(self, instance, modules: List[HXModule],
                 func: Optional[Callable]) -> None:
        """
        A module which wrappes a number of HXModules defined in `modules` to
        which a single PyTorch-differential function `func` is defined. For
        instance, this allows to wrap a Synapse and a Neuron to descripe
        recurrence.
        :param instance: The instance to register this wrapper in.
        :param modules: A list of modules to be represented by this wrapper.
        :param func: The function describing the unified functionallity of all
            modules assigned to this wrapper. As for HXModules, this needs to
            be a PyTorch-differentiable function and can be either an
            autograd.Function or a function defined by PyTorch operation. The
            signature of this function is expected as:
            1. All positional arguments of each function in `modules` appended
               in the order given in `modules`.
            2. All keywords arguments of each function in `modules`. If a
               keyword is occurred multiple times it is post-fixed `_i`, where
               i is an integered incremented with each occurrence.
            3. A keyword argument `hw_data` if hardware data is expected, which
               is a tuple holding the data for each module for which data is
               expected. The order is defined by `modules`.
            The function is expected to output a tensor or a tuple of tensors
            for each module in `modules`, that can be assigned to the output
            handle of the corresponding HXModule.
        """
        if isinstance(func, torch.autograd.function.FunctionMeta):
            raise TypeError(
                "Currently HXModuleWrappers do not accept "
                + "'torch.autograd.Function's as 'func'. If you want to use "
                + "an 'torch.autograd.Function' as 'func' you can wrap it "
                + "with a function providing the appropriate input signature "
                + "and return type.")
        super().__init__(instance, func)
        self.modules = modules
        self.update_args(modules)

    def contains(self, modules: List[HXModule]) -> bool:
        """
        Checks whether a list of modules `modules` is registered in the
        wrapper.
        :param modules: The modules for which to check if they are registered.
        :return: Returns a bool indicating whether `modules` are a subset.
        """
        return set(modules).issubset(set(self.modules))

    def update(self, modules: List[HXModule],
               func: Optional[Callable] = None):
        """
        Update the modules and the function in the wrapper.
        :param modules: The new modules to assign to the wrapper.
        :param func: The new function to represent the modules in the wrapper.
        """
        self.modules = modules
        self.update_args(modules)
        self.func = func

    def update_args(self, modules: List[HXModule]):
        """
        Gathers the args and kwargs of all modules in `modules` and renames
        keyword arguments that occur multiple times.
        :param modules: The modules represented by the wrapper.
        """
        # Update args
        self.extra_args = ()
        for module in modules:
            self.extra_args += module.extra_args
        # Update kwargs -> rename double
        keys = [k for module in modules for k in module.extra_kwargs.keys()]
        vals = [v for module in modules for v in module.extra_kwargs.values()]
        keys = [k + str(keys[:i].count(k) + 1) if keys.count(k) > 1 else k
                for i, k in enumerate(keys)]
        self.extra_kwargs = dict(zip(keys, vals))

    # pylint: disable=redefined-builtin
    def exec_forward(self, input: Tuple[TensorHandle],
                     output: Tuple[TensorHandle],
                     hw_map: Dict[grenade.PopulationDescriptor,
                                  Tuple[torch.Tensor]]) -> None:
        """
        Execute the the forward function of the wrapper. This method assigns
        each output handle in `output` their corresponding PyTorch tensors and
        adds the wrapper's `func` to the PyTorch graph.
        :param input: A tuple of the input handles where each handle
            corresponds to a certain module. The order is defined by `modules`.
            Note, a module can have multiple input handles.
        :param output: A tuole of output handles, each correspnding to one
            module. The order is defined by `modules`.
        :param hw_map: The hardware data map.
        """
        # Hw data for each module
        hw_data = tuple(
            hw_map.get(module.descriptor) for module in self.modules)
        # Concat input handles according to self.modules order
        inputs = tuple(handle.observable_state for handle in input)
        # Execute function
        output_tensors = self.func(inputs, hw_data=hw_data)
        # Check for have tuples
        if not isinstance(output_tensors, tuple):
            output_tensors = (output_tensors,)
        # We expect the same number of outputs as we have modules
        # TODO: Allow for multiple outputs per module
        assert len(output_tensors) == len(self.modules)
        # Assign output tensors
        for output_tensor, output_handle in zip(output_tensors, output):
            out = (output_tensor,) if not isinstance(output_tensor, tuple) \
                else output_tensor
            output_handle.put(*out)
