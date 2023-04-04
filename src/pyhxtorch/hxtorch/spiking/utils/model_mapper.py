"""
Mapper mixin to map Norse models to hxtorch models
"""
from typing import Any, Dict
import torch
import hxtorch


class MixinHXModelMapper:
    """ Mapper mixin to map snn models to hxtorch.spiking models """

    def map_from(self, state_dict: Any, mapping: Dict, map_all: bool = True):
        """
        Map an SNN model with state dict `state_dict` to this model.
        :param state_dict: The state dict used to map from.
        :param mapping: The mapping from the keys of this model's state dict to
            the keys in `state_dict`.
        :param map_all: A boolean value indicating if a state dict key is
            looked up in `state_dict` if not present in `mapping`. This allows
            to only define a mapping for some keys and load the others
            implicitly if the key names match. Defaults to true.
        """
        logger = hxtorch.logger.get("hxtorch.spiking.mapper")

        this_state = self.state_dict()
        logger.INFO(f"Model state dict:\n{state_dict.keys()}")

        for this_key in mapping.keys():
            if this_key not in this_state.keys():
                logger.WARN(
                    f"Model '{self.__class__.__name__}' does not have a "
                    + f"module named '{this_key}'. Possible module names are "
                    + f"{list(this_state.keys())}")

        for this_key in this_state.keys():
            if not map_all:
                if this_key not in mapping.keys():
                    continue
            if this_key in state_dict.keys():
                key = this_key
            elif this_key in mapping.keys():
                key = mapping[this_key]
            else:
                logger.ERROR(
                    f"This model does not have a key '{this_key}' or the "
                    + "source model state dict does not have a module named "
                    + f"'{key}'. Possible target keys are {this_state.keys()} "
                    + f"and possible source keys are {state_dict.keys()}")

            param = state_dict[key]
            if isinstance(param, torch.nn.parameter.Parameter):
                param = param.data
            this_state[this_key].copy_(param)
            logger.TRACE(
                f"Mapped key '{key}' to key '{this_key}'")

        logger.INFO("Model parameters restored from given model state dict.")
