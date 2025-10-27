"""
Generic parameter object holding hardware configurable neuron parameters.
"""
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

import torch


# pylint: disable=invalid-name
def equal(x: Union[torch.Tensor, float, int],
          y: Union[torch.Tensor, float, int]) -> bool:
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    return torch.allclose(x, y) and x.shape == y.shape


class HXBaseParameter:
    def __init__(self, hardware_value, model_value):
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

    def model_value_detach(self):
        return self._model_value

    def hardware_value_detach(self):
        return self._hardware_value

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"hardware_value={self.hardware_value}, "
            f"model_value={self.model_value})"
        )


class HXParameter(HXBaseParameter):
    def __init__(self, value: Any):
        super().__init__(value, value)

    @property
    def model_value(self):
        return self._hardware_value


class MixedHXModelParameter(HXBaseParameter):
    def __init__(self, model_value: Any, hardware_value: Any):
        super().__init__(hardware_value, model_value)


class HXTransformedModelParameter(HXBaseParameter):
    def __init__(self, model_value: Any, transform_func: Callable):
        super().__init__(None, model_value)
        self._func = transform_func

    @property
    def hardware_value(self):
        return self._func(self.model_value)

    @hardware_value.setter
    def hardware_value(self, hardware_value):
        self._hardware_value = hardware_value


class ModelParameter(HXBaseParameter):
    def __init__(self, model_value: Any):
        super().__init__(model_value, model_value)

    @property
    def hardware_value(self):
        return self._model_value

    @hardware_value.setter
    def hardware_value(self, hardware_value):
        self._hardware_value = hardware_value


# pylint: disable=super-init-not-called
class MockParameter(ModelParameter):
    def __init__(self, mean: Union[torch.Tensor, float, int],
                 std: Union[torch.Tensor, float, int, None] = None):
        """
        If mean and std are torch.Tensors, they must be of the same size or
        std must have only one element..
        """
        self._mean = mean
        self._std = std
        self.sample()

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        if not equal(self._mean, mean):
            self._mean = mean
            self.sample()

    @property
    def std(self):
        return self._std

    @std.setter
    def std(self, std):
        if not equal(self._std, std):
            self._std = std
            self.sample()

    def sample(self, size: Optional[int] = None):
        if self._std is None and size is None:
            self._model_value = self._mean
        elif self._std is None:
            self._model_value = torch.as_tensor(self._mean).expand(size)
        elif size is None:
            self._model_value = torch.normal(
                mean=torch.as_tensor(self._mean, dtype=torch.float32),
                std=torch.as_tensor(self._std, dtype=torch.float32))
        else:
            self._model_value = torch.normal(
                mean=torch.as_tensor(
                    self._mean, dtype=torch.float32).expand(size),
                std=torch.as_tensor(
                    self._std, dtype=torch.float32).expand(size))


ParameterType = Union[
    HXParameter, MixedHXModelParameter, HXTransformedModelParameter,
    ModelParameter, MockParameter]
