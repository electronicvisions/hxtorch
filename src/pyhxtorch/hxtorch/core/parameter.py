"""
Generic parameter object holding hardware configurable neuron parameters.
"""
from typing import Any, Union, Callable


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


ParameterType = Union[
    HXParameter, MixedHXModelParameter, HXTransformedModelParameter,
    ModelParameter]
