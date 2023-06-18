"""
Define module types
"""
import torch
from hxtorch.spiking.modules.hx_module import HXModule


# c.f.: https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class Population(HXModule):
    """ Base class for populations on BSS-2 """


# c.f.: https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class Projection(HXModule):
    """ Base class for projections on BSS-2 """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
