"""
Implementing a module wrapper to wrap multiple modules as one
"""
# pylint: disable=too-many-lines
from __future__ import annotations
from typing import TYPE_CHECKING, Union, Dict, Tuple, List, Optional
import pylogging as logger

import torch
import pygrenade_vx.network as grenade

from hxtorch.spiking.handle import TensorHandle
from hxtorch.spiking.modules.hx_module import HXFunctionalModule, HXModule
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment

log = logger.get("hxtorch.spiking.modules")


class HXModuleWrapper(HXFunctionalModule):  # pylint: disable=abstract-method
    """ Class to wrap HXModules """

    def __init__(self, experiment: Experiment, **modules: List[HXModule]) \
            -> None:
        """
        A module which wraps a number of HXModules defined in `modules` for
        which a single PyTorch-differential member function `forward_func` is
        defined. For instance, this allows to wrap a Synapse and a Neuron to
        describe recurrence.

        :param experiment: The experiment to register this wrapper in.
        :param modules: A list of modules to be represented by this wrapper.
        """
        super().__init__(experiment)
        self.modules = modules
        for name, module in modules.items():
            setattr(self, name, module)

    # Using input to be consistent with torch
    # pylint: disable=redefined-builtin
    def forward_func(self, input: TensorHandle,
                     hw_data: Optional[Tuple[torch.Tensor]] = None) \
            -> TensorHandle:
        """
        This function describes the unified functionality of all modules
        assigned to this wrapper. As for HXModules, this needs to be a PyTorch-
        differentiable function defined by PyTorch operations. The input and
        output of this member function is wrapped by (tuples of) `Handles`. The
        signature of this function is expected as:
        - Input: All input handles required for each module in `modules` as
            positional arguments in the order given by `modules`.
        - Outputs: Output a tuple of handles each corresponding to the output
            of one module in `modules`. The order is given by `modules`.
        - Additionally, hardware data can be accessed via a `hw_data` keyword
            arguments to which the the hardware data is supplied via a tuple
            holding the hardware data for each module.
        """
        return input

    def extra_repr(self) -> str:
        """ Add additional information """
        reprs = "modules=("
        for name, module in self.modules.items():
            reprs += f"\n\t{name}: {module}"
        reprs += ")\n, "
        reprs += f"{super().extra_repr()}"
        return reprs

    def contains(self, modules: Union[HXModule, List[HXModule]]) -> bool:
        """
        Checks whether a list of modules `modules` is registered in the
        wrapper.
        :param modules: The modules for which to check if they are registered.
        :return: Returns a bool indicating whether `modules` are a subset.
        """
        if not isinstance(modules, list):
            modules = [modules]
        for module in modules:
            if module not in self.modules.values():
                return False
        return True

    def update(self, **modules: Dict[HXModule]):
        """
        Update the modules and the function in the wrapper.
        :param modules: The new modules to assign to the wrapper.
        """
        self.modules = modules

    # pylint: disable=arguments-differ
    def forward(self):
        """ Forward method registering layer operation in given experiment """
        self.experiment.connect_wrapper(self)

    # pylint: disable=redefined-builtin
    def exec_forward(self, input: Tuple[TensorHandle],
                     output: Tuple[TensorHandle],
                     hw_map: Dict[
                         grenade.PopulationOnNetwork, Tuple[torch.Tensor]]) \
            -> None:
        """
        Execute the the forward function of the wrapper. This method assigns
        each output handle in `output` their corresponding PyTorch tensors and
        adds the wrapper's `forward_func` to the PyTorch graph.
        :param input: A tuple of the input handles where each handle
            corresponds to a certain module. The order is defined by `modules`.
            Note, a module can have multiple input handles.
        :param output: A tuple of output handles, each corresponding to one
            module. The order is defined by `modules`.
        :param hw_map: The hardware data map.
        """
        # Hw data for each module
        hw_data = tuple(
            hw_map.get(module.descriptor) for module in self.modules.values())
        # Concat input handles according to self.modules order
        output_tensors = self.func(input, hw_data=hw_data)
        # Check for have tuples
        if not isinstance(output_tensors, tuple):
            output_tensors = (output_tensors,)
        # We expect the same number of outputs as we have modules
        # TODO: Allow for multiple outputs per module
        assert len(output_tensors) == len(self.modules)
        # Assign output tensors
        for returned_handle, output_handle in zip(output_tensors, output):
            output_handle.clone(returned_handle)
