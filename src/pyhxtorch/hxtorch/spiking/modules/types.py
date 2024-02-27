"""
Define module types
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Union, Optional, Callable
import torch
from hxtorch.spiking.modules.hx_module import HXModule
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment
    from hxtorch.spiking.execution_instance import ExecutionInstance


# c.f.: https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class Population(HXModule):
    """ Base class for populations on BSS-2 """
    __constants__ = ['size']
    size: int

    def __init__(self, size: int, experiment: Experiment,
                 func: Union[Callable, torch.autograd.Function],
                 execution_instance: Optional[ExecutionInstance] = None) \
            -> None:
        """
        :param size: Number of input neurons.
        :param experiment: Experiment to append layer to.
        :param execution_instance: Execution instance to place to.
        :param func: Callable function implementing the module's forward
            functionality or a torch.autograd.Function implementing the
            module's forward and backward operation.
            TODO: Inform about func args
        """
        super().__init__(experiment, func, execution_instance)
        self.size = size

    def extra_repr(self) -> str:
        """ Add additional information """
        return f"size={self.size}, {super().extra_repr()}"


# c.f.: https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class Projection(HXModule):
    """ Base class for projections on BSS-2 """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    # pylint: disable=too-many-arguments
    def __init__(self, in_features: int, out_features: int,
                 experiment: Experiment,
                 func: Union[Callable, torch.autograd.Function],
                 execution_instance: ExecutionInstance) -> None:
        """
        :param experiment: Experiment to append layer to.
        :param in_features: Size of input dimension.
        :param out_features: Size of output dimension.
        :param experiment: Experiment to append layer to.
        :param func: Callable function implementing the module's forward
            functionality or a torch.autograd.Function implementing the
            module's forward and backward operation.
            TODO: Inform about func args
        :param execution_instance: Execution instance to place to.
        """
        super().__init__(experiment, func, execution_instance)
        self.in_features = in_features
        self.out_features = out_features

    def extra_repr(self) -> str:
        """ Add additional information """
        reprs = f"in_features={self.in_features}, "
        reprs += f"out_features={self.out_features}, "
        reprs += f"{super().extra_repr()}"
        return reprs
