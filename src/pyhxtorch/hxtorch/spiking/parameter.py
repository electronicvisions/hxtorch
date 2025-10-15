"""
Generic parameter object holding hardware configurable neuron parameters.
"""
from typing import Any, Callable
import torch

from hxtorch.core.parameter import (
    HXParameter,
    MixedHXModelParameter,
    HXTransformedModelParameter,
    ModelParameter,
)


class TrainableParameterMixin(torch.nn.Module):
    def __init__(self, set_on_hw_func: Callable = None):
        torch.nn.Module.__init__(self)
        self.make_trainable(set_on_hw_func=set_on_hw_func)

    def model_value_detach(self):
        if self.is_trainable():
            return self._model_value.detach()
        return self._model_value

    def hardware_value_detach(self):
        if self.is_trainable():
            if torch.is_tensor(self._hardware_value):
                return self._hardware_value.detach()
            return self._hardware_value
        return self._hardware_value

    def set_hw_config(self, neuron_coordinates, neuron_configs):
        if self.set_on_hw_func is None:
            raise ValueError(
                'When executing on HW,'
                + 'set_on_hw_func needs to be provided.'
            )
        self.set_on_hw_func(
            self.hardware_value,
            neuron_configs,
            neuron_coordinates,
        )

    def is_trainable(self):
        return isinstance(self._model_value, torch.nn.Parameter)

    def make_trainable(self, set_on_hw_func=None):
        self.set_on_hw_func = set_on_hw_func

        if not torch.is_tensor(self._model_value):
            self._model_value = torch.tensor(
                self._model_value
            )

        # For transformed parameters, _hardware_value may still be None at this
        # point; use the property, which computes it from model_value.
        hardware_value = self._hardware_value
        if hardware_value is None:
            hardware_value = self.hardware_value

        # Keep structured values (e.g. tuple of tensors) unchanged.
        if torch.is_tensor(hardware_value):
            self._hardware_value = hardware_value
        else:
            self._hardware_value = torch.tensor(hardware_value)

        self._model_value = torch.nn.Parameter(
            self._model_value,
            requires_grad=True,
        )
        return self

    def forward(self):
        return self._model_value

    def __str__(self):
        return f"{self.__class__.__name__}(at ({id(self)}), hardware_value=" \
            + f"{self.hardware_value}, model_value={self.model_value}, " \
            + f"trainable={self.is_trainable()})"


class TrainableHXParameter(
    TrainableParameterMixin,
    HXParameter,
):
    def __init__(
        self,
        value: Any,
        set_on_hw_func: Callable = None,
    ):
        HXParameter.__init__(
            self,
            value,
        )
        TrainableParameterMixin.__init__(
            self,
            set_on_hw_func=set_on_hw_func
        )


class TrainableMixedHXModelParameter(
    TrainableParameterMixin,
    MixedHXModelParameter,
):
    def __init__(
        self,
        model_value: Any,
        hardware_value: Any,
        set_on_hw_func: Callable = None,
    ):
        MixedHXModelParameter.__init__(
            self,
            model_value,
            hardware_value,
        )
        TrainableParameterMixin.__init__(
            self,
            set_on_hw_func=set_on_hw_func,
        )


class TrainableHXTransformedModelParameter(
    TrainableParameterMixin,
    HXTransformedModelParameter,
):
    def __init__(
        self,
        model_value: Any,
        transform_func: Callable,
        set_on_hw_func: Callable = None,
    ):
        HXTransformedModelParameter.__init__(
            self,
            model_value,
            transform_func,
        )
        TrainableParameterMixin.__init__(
            self,
            set_on_hw_func=set_on_hw_func,
        )


class TrainableModelParameter(
    TrainableParameterMixin,
    ModelParameter,
):
    def __init__(
        self,
        model_value: Any,
        set_on_hw_func: Callable = None,
    ):
        ModelParameter.__init__(
            self,
            model_value,
        )
        TrainableParameterMixin.__init__(
            self,
            set_on_hw_func=set_on_hw_func,
        )
