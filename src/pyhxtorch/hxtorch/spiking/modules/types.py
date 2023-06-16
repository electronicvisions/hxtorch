"""
Define module types
"""
import torch
from hxtorch.spiking.modules.hx_module import HXModule


class Population(HXModule):
    """ Base class for populations on BSS-2 """
    # pylint: disable=abstract-method
    pass


class Projection(HXModule):
    """ Base class for projections on BSS-2 """
    # pylint: disable=abstract-method
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
