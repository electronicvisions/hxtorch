"""
Define module types
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from hxtorch.core.modules.projection import Projection as CoreProjection
from hxtorch.spiking.modules.hx_module import HXTorchBaseModule
from hxtorch.spiking.handle import SynapseHandle

if TYPE_CHECKING:
    from hxtorch.core.modules.population import Population as BasePopulation


# c.f.: https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class Projection(HXTorchBaseModule, CoreProjection):
    """ Base class for projections on BSS-2 """
    weight: torch.Tensor
    output_type = SynapseHandle

    def extra_repr(self) -> str:
        """ Add additional information """
        reprs = f"in_features={self.in_features}, "
        reprs += f"out_features={self.out_features}, "
        reprs += f"{super().extra_repr()}"
        return reprs

    def __init__(self, *prj_args, **prj_kwargs) -> None:
        CoreProjection.__init__(self, *prj_args, **prj_kwargs)
        HXTorchBaseModule.__init__(self)

    def source_population(self) -> BasePopulation:
        return self.experiment.modules.source_populations(self).pop()

    def target_population(self) -> BasePopulation:
        return self.experiment.modules.target_populations(self).pop()

    def post_process(self, *args, **kwargs) -> None:
        return None
