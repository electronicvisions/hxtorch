"""
Generic parameter object holding hardware configurable neuron parameters.
"""
from typing import Union, Callable
import torch


class HXBaseParameter(torch.nn.Module):
    def __init__(self, hardware_value, model_value):
        super().__init__()
        self._hardware_value = hardware_value
        self._model_value = model_value
        self.set_on_chip_func = None

    @property
    def hardware_value(self):
        return self._hardware_value

    @hardware_value.setter
    def hardware_value(self, hardware_value):
        self._hardware_value = hardware_value

    @property
    def model_value(self):
        return self._model_value

    @model_value.setter
    def model_value(self, model_value):
        self._model_value = model_value

    def set_on_chip(self, chip, neuron_coordinates):
        if self.set_on_chip_func is None:
            raise ValueError(
                'When executing on HW,'
                + 'set_on_chip_func needs to be provided.'
            )
        self.set_on_chip_func(self.hardware_value, chip,
                              neuron_coordinates)

    def is_trainable(self):
        return isinstance(self._model_value, torch.nn.Parameter)

    def make_trainable(self, set_on_chip_func=None):
        self.set_on_chip_func = set_on_chip_func
        if not torch.is_tensor(self._model_value):
            self._model_value = torch.tensor(
                self._model_value
            )
        self._model_value = torch.nn.Parameter(
            self._model_value
        )
        return self

    def forward(self):
        return self._model_value

    def __str__(self):
        return f"{self.__class__.__name__}(at ({id(self)}), hardware_value=" \
            + f"{self.hardware_value}, model_value={self.model_value}, " \
            + f"trainable={self.is_trainable()})"


class HXParameter(HXBaseParameter):
    def __init__(self, value: Union[torch.Tensor, float, int]):
        super().__init__(value, value)

    @property
    def model_value(self):
        return self._hardware_value


class MixedHXModelParameter(HXBaseParameter):
    def __init__(self, model_value: Union[torch.Tensor, float, int],
                 hardware_value: Union[torch.Tensor, float, int]):
        super().__init__(hardware_value, model_value)


class HXTransformedModelParameter(HXBaseParameter):
    def __init__(self, model_value: Union[torch.Tensor, float, int],
                 transform_func: Callable):
        super().__init__(None, model_value)
        self._func = transform_func

    @property
    def hardware_value(self):
        return self._func(self.model_value)

    @hardware_value.setter
    def hardware_value(self, hardware_value):
        self._hardware_value = hardware_value


class ModelParameter(HXBaseParameter):
    def __init__(self, model_value: Union[torch.Tensor, float, int]):
        super().__init__(model_value, model_value)

    @property
    def hardware_value(self):
        return self._model_value

    @hardware_value.setter
    def hardware_value(self, hardware_value):
        self._hardware_value = hardware_value


ParameterType = Union[
    HXParameter, MixedHXModelParameter, HXTransformedModelParameter,
    ModelParameter]
