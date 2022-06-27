"""
Defining basic types to create hw-executable instances
"""
from typing import Dict, Tuple
from collections import OrderedDict

import torch
import hxtorch
from hxtorch.snn.handle import TensorHandle

log = hxtorch.logger.get("hxtorch.snn.instance")


class Instance:

    """ Instance class for describing experiments on hardware """

    def __init__(self, mock: bool = True) -> None:
        """
        Instanziate a new instance, represting an experiment on hardware and/or
        in software.

        :param mock: Indicating whether module is executed on hardware (False)
            or simulated in software (True).
        """
        self.mock = mock
        self.connections = dict()

    def connect(self, module: torch.nn.Module, input_handle: TensorHandle,
                output_handle: TensorHandle) -> None:
        """
        Add an module to the instance and connect it to other instance
        modules via input and output handles.

        :param module: The HXModule to add to the instance.
        :param input_handle: The HXTensorHandle serving as input to the module
            (its obsv_state).
        :param output_handle: The HXTensorHandle outputted by the module,
            serving as input to subsequent HXModules.
        """
        self.connections.update({module: (input_handle, output_handle)})

    def sorted(self) -> Dict[
            torch.nn.Module, Tuple[TensorHandle, TensorHandle]]:
        """
        Sort the registeres modules according to according to the corresponding
        data flow indicated by the input and output handles.

        FIXME: Ugly. Probably wrong for recurrent connections.

        :return: Returns an ordered dict, with an order in line with the data
            flow.
        """
        # Initialize sorted list with first element
        sorted_modules = OrderedDict()
        outputs = [o for (_, o) in self.connections.values()]
        inputs = [i for (i, _) in self.connections.values()]

        while len(outputs) != 0:
            for module, (input_handle, output_handle) in \
                    self.connections.items():
                assert input_handle != output_handle
                if input_handle not in outputs and input_handle in inputs:
                    assert output_handle in outputs
                    sorted_modules[module] = (input_handle, output_handle)
                    outputs.remove(output_handle)
                    inputs.remove(input_handle)

        return sorted_modules
