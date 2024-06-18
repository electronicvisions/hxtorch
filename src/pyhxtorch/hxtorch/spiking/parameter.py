"""
Generic parameter object holding hardware configurable neuron parameters.
"""
from typing import Union, Callable
import torch


class HXBaseParameter:

    def __init__(self, hardware_value, model_value):
        self._hardware_value = hardware_value
        self._model_value = model_value

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

    def __str__(self):
        return f"{self.__class__.__name__}(at ({id(self)}), hardware_value=" \
            + f"{self.hardware_value}, model_value={self.model_value})"


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
