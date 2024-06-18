"""
Define module types
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, List, Union
import numpy as np
import quantities as pq
import torch

from dlens_vx_v3 import halco

from hxtorch.spiking.modules.hx_module import HXModule
from hxtorch.spiking.parameter import (
    HXBaseParameter, HXParameter, ParameterType)

if TYPE_CHECKING:
    from calix.spiking.neuron import SpikingCalibTarget
    from hxtorch.spiking.experiment import Experiment
    from hxtorch.spiking.execution_instance import ExecutionInstance


class BasePopulation(HXModule):
    """ Base class for populations """
    __constants__ = ['size']
    size: int

    def __init__(self, size: int, experiment: Experiment,
                 execution_instance: Optional[ExecutionInstance] = None) \
            -> None:
        """
        :param size: Number of input neurons.
        :param experiment: Experiment to append layer to.
        :param execution_instance: Execution instance to place to.
        """
        super().__init__(experiment, execution_instance)
        self.size = size

    def extra_repr(self) -> str:
        """ Add additional information """
        return f"size={self.size}, {super().extra_repr()}"


class InputPopulation(BasePopulation):
    """ Base type for external input populations """


ModuleParameterType = Union[ParameterType, torch.Tensor, float, int]


# c.f.: https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class Population(BasePopulation):
    """ Base class for on-chip populations on BSS-2 """
    __constants__ = ['size']
    size: int

    _calibratable_parameters_defaults = {  # pylint: disable=invalid-name
        "leak": 80,
        "reset": 80,
        "threshold": 125,
        "tau_mem": 1e-5,
        "tau_syn": 1e-5,
        "i_synin_gm": 500,
        "membrane_capacitance": 63,
        "refractory_time": 2e-6,
        "synapse_dac_bias": 600,
        "holdoff_time": 0,
        "e_coba_reversal": None,
        "e_coba_reference": None
    }

    def __init__(self, size: int, experiment: Experiment,
                 execution_instance: Optional[ExecutionInstance] = None,
                 **hxparams: Dict[str, ModuleParameterType]) \
            -> None:
        """
        :param size: Number of input neurons.
        :param experiment: Experiment to append layer to.
        :param execution_instance: Execution instance to place to.
        """
        super().__init__(size, experiment, execution_instance)
        self.unit_ids = None
        self._generate_hxparameters(hxparams)
        self.read_params_from_calibration = True
        self._params_hash = None

    def _generate_hxparameters(self, hxparams: Dict[str, ModuleParameterType]):
        for param_key, param_default in \
                self._calibratable_parameters_defaults.items():
            try:
                param = hxparams[param_key]
            except KeyError:
                param = param_default
            if not isinstance(param, HXBaseParameter):
                param = HXParameter(param)
            setattr(self, param_key, param)

    def extra_repr(self) -> str:
        reprs = ""
        for key, value in self.params_dict().items():
            reprs += f"{key}={value}, "
        reprs += f"{super().extra_repr()}"
        return reprs

    def params_dict(self) -> Dict:
        return {param: getattr(self, param)
                for param in self._calibratable_parameters_defaults}

    def calib_changed_since_last_run(self) -> bool:
        new_params_hash = hash(frozenset(self.params_dict()))
        calibrate = self._params_hash != new_params_hash
        self._params_hash = new_params_hash
        return calibrate

    # pylint: disable=too-many-branches
    def params_from_calibration(self, targets: SpikingCalibTarget, neurons) \
            -> None:
        if not self.read_params_from_calibration:
            return

        # Select only neuron calib for population
        targets = targets.neuron_target

        size = len(neurons)
        coords = self._get_an_indices(neurons)
        mapping = self._get_an_to_in_pop_indices(neurons)
        selector = self._get_in_pop_indices(neurons)
        assert len(coords) == len(mapping)
        assert len(selector) == size

        # NOTE: We demand that all AtomicNeurons of the same LogicalNeuron do
        #       have the same values. This might change in the future if multi-
        #       compartment neurons are introduced.
        def assert_same_values_per_neuron(entity: torch.Tensor):
            if len(entity.shape) > 1:
                for row_values in entity:
                    return assert_same_values_per_neuron(row_values)
            assert all(
                torch.all(nrn_values[0] == nrn_values)
                for nrn_values in [
                    torch.tensor([entity[coords[mapping == i]]])
                    for i in range(size)]) or all(
                        torch.isnan(torch.tensor(entity[coords])))
            return None

        # i_synin_gm
        self.i_synin_gm.hardware_value = torch.tensor(targets.i_synin_gm) \
            * torch.ones(2, dtype=torch.int64)

        # synapse_dac_bias
        self.synapse_dac_bias.hardware_value = torch.tensor(
            targets.synapse_dac_bias)

        # tau_mem, refractory_time, holdoff_time
        # leak, reset, threshold, membrane_capacitance
        for key in ["tau_mem", "refractory_time", "holdoff_time",
                    "leak", "reset", "threshold", "membrane_capacitance"]:
            value = getattr(targets, key)
            if isinstance(value, pq.Quantity):
                value = value.rescale(pq.s)
            value = torch.tensor(value)
            if not value.shape:  # Single number
                getattr(self, key).hardware_value = torch.full(
                    (halco.AtomicNeuronOnDLS.size,), value)[selector]
            if len(value.shape) == 1:  # 1D
                assert_same_values_per_neuron(value)
                getattr(self, key).hardware_value = value[selector]

        # tau_syn
        if (not isinstance(targets.tau_syn, np.ndarray)
                or not targets.tau_syn.shape):
            self.tau_syn.hardware_value = torch.full(
                (halco.AtomicNeuronOnDLS.size,),
                torch.tensor(targets.tau_syn.rescale(pq.s)))[selector]
        elif targets.tau_syn.shape == (halco.AtomicNeuronOnDLS.size,):
            assert_same_values_per_neuron(targets.tau_syn)
            self.tau_syn.hardware_value = torch.tensor(
                targets.tau_syn.rescale(pq.s)).repeat(2, 1)[:, selector]
        elif targets.tau_syn.shape == (halco.SynapticInputOnNeuron.size,):
            self.tau_syn.hardware_value = torch.tensor(
                targets.tau_syn.rescale(pq.s)).repeat(
                    1, halco.AtomicNeuronOnDLS.size)[:, selector]
        elif targets.tau_syn.shape == (halco.SynapticInputOnNeuron.size,
                                       halco.AtomicNeuronOnDLS.size):
            assert_same_values_per_neuron(torch.tensor(targets.tau_syn))
            self.tau_syn.hardware_value = torch.tensor(
                targets.tau_syn.rescale(pq.s))[:, selector]

        # e_coba_reversal, e_coba_reference
        for key in ["e_coba_reversal", "e_coba_reference"]:
            value = getattr(targets, key)
            if value is None:
                setattr(self, key, value)
            elif value.shape == (halco.SynapticInputOnNeuron.size,):
                setattr(self, key, torch.tensor(value).repeat(
                        (1, halco.AtomicNeuronOnDLS.size))[:, selector])
            elif value.shape == (halco.SynapticInputOnNeuron.size,
                                 halco.AtomicNeuronOnDLS.size):
                assert_same_values_per_neuron(value)
                setattr(self, key, torch.tensor(value)[:, selector])

        self._params_hash = hash(frozenset(self.params_dict()))

    def calibration_from_params(self, targets: SpikingCalibTarget, neurons) \
            -> SpikingCalibTarget:
        """
        Add population specific calibration targets to the experiment-wide
        calibration target, which holds information for all populations.

        :param spiking_calib_target: Calibration target parameters of all
            neuron populations registered in the self.experiment instance.
        :returns: The chip_wide_calib_target with adjusted parameters.
        """
        # Select only neuron calib for population
        targets = targets.neuron_target

        size = len(neurons)
        coords = self._get_an_indices(neurons)
        mapping = self._get_an_to_in_pop_indices(neurons)
        assert len(coords) == len(mapping)

        # i_synin_gm
        i_synin_gm = np.array(self.i_synin_gm.hardware_value)
        if not i_synin_gm.shape:
            i_synin_gm = i_synin_gm * np.ones(2, dtype=np.int64)
        for i in range(2):
            if (targets.i_synin_gm[i] is not None
                    and targets.i_synin_gm[i] != i_synin_gm[i]):
                raise AttributeError(
                    f"'i_synin_gm[{i}]' requires same value for all neurons")
        targets.i_synin_gm = i_synin_gm

        # synapse_dac_bias
        if (targets.synapse_dac_bias is not None
                and self.synapse_dac_bias.hardware_value
                != targets.synapse_dac_bias):
            raise AttributeError(
                "'synapse_dac_bias' requires same value for all neurons")
        targets.synapse_dac_bias = int(self.synapse_dac_bias.hardware_value)

        # tau_syn
        targets.tau_syn[:, coords] = (
            self._resize(self.tau_syn.hardware_value, size, rows=2).numpy()
            * pq.s).rescale(pq.us)[:, mapping]

        # e_coba_reversal
        if self.e_coba_reversal.hardware_value is None:
            targets.e_coba_reversal[:, coords] = np.repeat(
                np.array([np.inf, -np.inf])[:, np.newaxis],
                size, axis=1)[:, mapping]
        else:
            targets.e_coba_reversal[:, coords] = self._resize(
                self.e_coba_reversal.hardware_value, size,
                rows=2).numpy()[:, mapping]

        # e_coba_reference
        if self.e_coba_reference.hardware_value is None:
            targets.e_coba_reference[:, coords] = np.ones((
                halco.SynapticInputOnNeuron.size, size))[:, mapping] * np.nan
        else:
            targets.e_coba_reference[:, coords] = self._resize(
                self.e_coba_reference.hardware_value, size,
                rows=2).numpy()[:, mapping]

        # tau_mem, refractory_time, holdoff_time
        for key in ["tau_mem", "refractory_time", "holdoff_time"]:
            getattr(targets, key)[coords] = (
                self._resize(getattr(self, key).hardware_value, size).numpy()
                * pq.s).rescale(pq.us)[mapping]

        # leak, reset, threshold, membrane_capacitance
        for key in ["leak", "reset", "threshold", "membrane_capacitance"]:
            getattr(targets, key)[coords] = self._resize(
                getattr(self, key).hardware_value, size).numpy()[mapping]
        return targets

    # pylint: disable=too-many-return-statements
    @staticmethod
    def _resize(entity: torch.tensor, size: int, rows: int = 1):
        if not isinstance(entity, torch.Tensor):
            entity = torch.tensor(entity)
        if rows == 1:
            if not entity.shape:
                return torch.full((size,), entity)
            if entity.shape[0] == 1:
                return entity.clone().detach().repeat(size)
            return entity.clone().detach()
        assert rows == 2
        if not entity.shape:
            return torch.full((2, size), entity)
        if len(entity.shape) == 1:
            if entity.size == 2:
                return entity.repeat((1, size)).clone().detach()
            if entity.size == halco.AtomicNeuronOnDLS.size:
                return entity.repeat((2, 1)).clone().detach()
        return entity.clone().detach()

    @staticmethod
    def _get_an_indices(neurons: List[halco.LogicalNeuronOnDLS]):
        return [an.toEnum().value() for neuron in neurons
                for an in neuron.get_atomic_neurons()]

    @staticmethod
    def _get_in_pop_indices(neurons: List[halco.LogicalNeuronOnDLS]):
        return [neuron.get_atomic_neurons()[0].toEnum().value()
                for neuron in neurons]

    @staticmethod
    def _get_an_to_in_pop_indices(neurons: List[halco.LogicalNeuronOnDLS]):
        return [in_pop_id for in_pop_id, neuron in enumerate(neurons)
                for _ in neuron.get_atomic_neurons()]


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
                 execution_instance: ExecutionInstance) -> None:
        """
        :param experiment: Experiment to append layer to.
        :param in_features: Size of input dimension.
        :param out_features: Size of output dimension.
        :param experiment: Experiment to append layer to.
        :param execution_instance: Execution instance to place to.
        """
        super().__init__(experiment, execution_instance)
        self.in_features = in_features
        self.out_features = out_features

    def extra_repr(self) -> str:
        """ Add additional information """
        reprs = f"in_features={self.in_features}, "
        reprs += f"out_features={self.out_features}, "
        reprs += f"{super().extra_repr()}"
        return reprs
