"""
Define module types
"""
from __future__ import annotations

import torch

from hxtorch.core.modules.population import (
    InputPopulation as CoreInputPopulation
)
from hxtorch.core.modules.population import Population as CorePopulation
from hxtorch.spiking.modules.hx_module import HXTorchBaseModule
from hxtorch.spiking.handle import Handle
from hxtorch.spiking.observables import HXTorchObservables


class InputPopulation(HXTorchBaseModule, CoreInputPopulation):
    """ Base type for external input populations """
    output_type = type(Handle('spikes'))
    _observables_factory = HXTorchObservables

    def __init__(self, *pop_args, **pop_kwargs) -> None:
        HXTorchBaseModule.__init__(self)
        CoreInputPopulation.__init__(self, *pop_args, **pop_kwargs)


# c.f.: https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class Population(HXTorchBaseModule, CorePopulation):
    _observables_factory = HXTorchObservables

    def __init__(self, *pop_args, **pop_kwargs) -> None:
        HXTorchBaseModule.__init__(self)
        CorePopulation.__init__(self, *pop_args, **pop_kwargs)
        self.read_params_from_calibration = True

    def extra_repr(self) -> str:
        reprs = f"size={self.size}, "
        for key, value in self.params_dict().items():
            reprs += f"{key}={value}, "
        reprs += f"{super().extra_repr()}"
        return reprs

    @staticmethod
    def resize_parameter_value(neuron_id, size, param):
        if param is None:
            return None
        val = param.hardware_value
        if (isinstance(val, torch.Tensor)
                and val.ndim > 0
                and val.shape[0] == size):
            assert val.ndim < 2
            return val[neuron_id]
        elif (isinstance(val, torch.Tensor)
              and val.numel() == 1):
            return val.item()
        elif isinstance(val, torch.Tensor):
            raise ValueError(
                f"Parameter value size {tuple(val.shape)} does not match "
                + "expected size "
                f"{size}."
            )
        return val
