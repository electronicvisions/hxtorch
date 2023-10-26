"""
Define module types
"""
from typing import (
    TYPE_CHECKING, Callable, Union)
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

    def __init__(self, experiment: "Experiment",
                 func: Union[Callable, torch.autograd.Function],
                 execution_instance: grenade.common.ExecutionInstanceID) \
            -> None:
        """
        :param experiment: Experiment to append layer to.
        :param execution_instance: Execution instance to place to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation.
            TODO: Inform about func args
        """
        HXModule.__init__(self, experiment, func)
        EntityOnExecutionInstance.__init__(self, execution_instance)


# c.f.: https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class Projection(HXModule, EntityOnExecutionInstance):
    """ Base class for projections on BSS-2 """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, experiment: "Experiment",
                 func: Union[Callable, torch.autograd.Function],
                 execution_instance: grenade.common.ExecutionInstanceID) \
            -> None:
        """
        :param experiment: Experiment to append layer to.
        :param execution_instance: Execution instance to place to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation.
            TODO: Inform about func args
        """
        HXModule.__init__(self, experiment, func)
        EntityOnExecutionInstance.__init__(self, execution_instance)
