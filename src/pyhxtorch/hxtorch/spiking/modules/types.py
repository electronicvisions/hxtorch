"""
Define module types
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Union
import torch
import pygrenade_vx as grenade
from hxtorch.spiking.modules.hx_module import HXModule
from hxtorch.spiking.modules.entity_on_execution_instance import \
    EntityOnExecutionInstance
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment


# c.f.: https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class Population(HXModule, EntityOnExecutionInstance):
    """ Base class for populations on BSS-2 """
    __constants__ = ['size']
    size: int

    def __init__(self, size: int, experiment: Experiment,
                 func: Union[Callable, torch.autograd.Function],
                 execution_instance: grenade.common.ExecutionInstanceID) \
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
        HXModule.__init__(self, experiment, func)
        EntityOnExecutionInstance.__init__(self, execution_instance)
        self.size = size

    def extra_repr(self) -> str:
        """ Add additional information """
        reprs = f"execution_instance={self.execution_instance}, "
        reprs += f"size={self.size}, {super().extra_repr()}"
        return reprs


# c.f.: https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class Projection(HXModule, EntityOnExecutionInstance):
    """ Base class for projections on BSS-2 """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    # pylint: disable=too-many-arguments
    def __init__(self, in_features: int, out_features: int,
                 experiment: Experiment,
                 func: Union[Callable, torch.autograd.Function],
                 execution_instance: grenade.common.ExecutionInstanceID) \
            -> None:
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
        HXModule.__init__(self, experiment, func)
        EntityOnExecutionInstance.__init__(self, execution_instance)
        self.in_features = in_features
        self.out_features = out_features

    def extra_repr(self) -> str:
        """ Add additional information """
        reprs = f"execution_instance={self.execution_instance}, "
        reprs += f"in_features={self.in_features}, "
        reprs += f"out_features={self.out_features}, "
        reprs += f"{super().extra_repr()}"
        return reprs
