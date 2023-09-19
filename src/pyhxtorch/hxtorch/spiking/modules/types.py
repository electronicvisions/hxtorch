"""
Define module types
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Union, Dict, Optional
import torch
from hxtorch.spiking.modules.hx_module import HXModule
if TYPE_CHECKING:
    from calix.spiking import SpikingCalibTarget
    from hxtorch.spiking.experiment import Experiment
    from hxtorch.spiking.execution_instance import ExecutionInstance


# c.f.: https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class Population(HXModule):
    """ Base class for populations on BSS-2 """
    __constants__ = ['size']
    size: int

    def __init__(self, size: int, experiment: Experiment,
                 func: Union[Callable, torch.autograd.Function],
                 execution_instance: Optional[ExecutionInstance] = None) \
            -> None:
        """
        :param size: Number of input neurons.
        :param experiment: Experiment to append layer to.
        :param execution_instance: Execution instance to place to.
        :param func: Callable function implementing the module's forward
            functionality or a torch.autograd.Function implementing the
            module's forward and backward operation.
            TODO: Inform about func args
        """
        super().__init__(experiment, func, execution_instance)
        self.size = size
        self.read_params_from_calibration = True
        self.params = None
        self._params_hash = None

    def extra_repr(self) -> str:
        """ Add additional information """
        return f"size={self.size}, {super().extra_repr()}"

    def calib_changed_since_last_run(self) -> bool:
        if not hasattr(self, "params"):
            return False
        new_params_hash = hash(self.params)
        calibrate = self._params_hash != new_params_hash
        self._params_hash = new_params_hash
        return calibrate

    def params_from_calibration(
            self, spiking_calib_target: SpikingCalibTarget) -> None:
        if hasattr(self, "params"):
            # Create a hash in each case, otherwise
            # calib_changed_since_last_run gets triggered
            self._params_hash = hash(self.params)
        if (not self.read_params_from_calibration
                or not hasattr(self, "params")
                or not hasattr(self.params, "from_calix_targets")):
            return
        # get populations HW neurons
        neurons = self.execution_instance.neuron_placement.id2logicalneuron(
            self.unit_ids)
        self.params = self.params.from_calix_targets(
            spiking_calib_target.neuron_target, neurons)
        self._params_hash = hash(self.params)
        # get params from calib target
        self.extra_kwargs.update({"params": self.params})

    def calibration_from_params(
            self, spiking_calib_target: SpikingCalibTarget) -> Dict:
        """
        Add population specific calibration targets to the experiment-wide
        calibration target, which holds information for all populations.

        :param spiking_calib_target: Calibration target parameters of all
            neuron populations registered in the self.experiment instance.
        :returns: The chip_wide_calib_target with adjusted parameters.
        """
        neurons = self.execution_instance.neuron_placement.id2logicalneuron(
            self.unit_ids)
        return self.params.to_calix_targets(
            spiking_calib_target.neuron_target, neurons)


# c.f.: https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class Projection(HXModule):
    """ Base class for projections on BSS-2 """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    # pylint: disable=too-many-arguments
    def __init__(self, in_features: int, out_features: int,
                 experiment: Experiment,
                 func: Union[Callable, torch.autograd.Function],
                 execution_instance: ExecutionInstance) -> None:
        """
        :param experiment: Experiment to append layer to.
        :param in_features: Size of input dimension.
        :param out_features: Size of output dimension.
        :param experiment: Experiment to append layer to.
        :param func: Callable function implementing the module's forward
            functionality or a torch.autograd.Function implementing the
            module's forward and backward operation.
            TODO: Inform about func args
        :param execution_instance: Execution instance to place to.
        """
        super().__init__(experiment, func, execution_instance)
        self.in_features = in_features
        self.out_features = out_features

    def extra_repr(self) -> str:
        """ Add additional information """
        reprs = f"in_features={self.in_features}, "
        reprs += f"out_features={self.out_features}, "
        reprs += f"{super().extra_repr()}"
        return reprs
