"""
Implementing BatchDropout Module
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Type, Optional, Union
import pylogging as logger

import torch

import hxtorch.spiking.functional as F
from hxtorch.spiking.handle import NeuronHandle
from hxtorch.spiking.modules.hx_module import HXModule
if TYPE_CHECKING:
    from hxtorch.spiking.experiment import Experiment

log = logger.get("hxtorch.spiking.modules")


class BatchDropout(HXModule):  # pylint: disable=abstract-method
    """
    Batch dropout layer

    Caveat:
    In-place operations on TensorHandles are not supported. Must be placed
    after a neuron layer, i.e. Neuron.
    """

    output_type: Type = NeuronHandle

    def __init__(self, size: int, dropout: float, experiment: Experiment,
                 func: Union[
                     Callable, torch.autograd.Function] = F.batch_dropout) \
            -> None:
        """
        Initialize BatchDropout layer. This layer disables spiking neurons in
        the previous spiking Neuron layer with a probability of `dropout`.
        Note, `size` has to be equal to the size in the corresponding spiking
        layer. The spiking mask is maintained for the whole batch.

        :param size: Size of the population this dropout layer is applied to.
        :param dropout: Probability that a neuron in the precessing layer gets
            disabled during training.
        :param experiment: Experiment to append layer to.
        :param func: Callable function implementing the module's forward
            functionallity or a torch.autograd.Function implementing the
            module's forward and backward operation. Defaults to
            `batch_dropout`.
        """
        super().__init__(experiment=experiment, func=func)

        self.size = size
        self._dropout = dropout
        self._mask: Optional[torch.Tensor] = None

    def set_mask(self) -> None:
        """
        Creates a new random dropout mask, applied to the spiking neurons in
        the previous module.
        If `module.eval()` dropout will be disabled.

        :returns: Returns a random boolen spike mask of size `self.size`.
        """
        if self.training:
            self.mask = (torch.rand(self.size) > self._dropout)
        else:
            self.mask = torch.ones(self.size).bool()
        self.extra_args = (self._mask,)

        return self._mask

    @property
    def mask(self) -> None:
        """
        Getter for spike mask.

        :returns: Returns the current spike mask.
        """
        return self._mask

    @mask.setter
    def mask(self, mask: torch.Tensor) -> None:
        """
        Setter for the spike mask.

        :param mask: Spike mask. Must be of shape `(self.size,)`.
        """
        # Mark dirty
        self._changed_since_last_run = True
        self._mask = mask
