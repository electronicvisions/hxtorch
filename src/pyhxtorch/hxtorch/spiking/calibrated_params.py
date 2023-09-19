"""
Generic parameter object holding hardware configurable neuron parameters.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Union, Optional
import dataclasses
import pylogging as logger
import numpy as np
import quantities as pq
import torch
from dlens_vx_v3 import halco

if TYPE_CHECKING:
    from calix.spiking.neuron import NeuronCalibTarget


@dataclasses.dataclass(unsafe_hash=True)
class CalibratedParams:
    """
    Parameters for any (of currently available) forward and backward path.
    """
    # calix's neuron-calibrateable parameters
    leak: torch.Tensor = 80
    reset: torch.Tensor = 80
    threshold: torch.Tensor = 125
    tau_mem: torch.Tensor = 1e-5
    tau_syn: torch.Tensor = 1e-5
    i_synin_gm: Union[int, torch.Tensor] = 500
    e_coba_reversal: Optional[torch.Tensor] = None
    e_coba_reference: Optional[torch.Tensor] = None
    membrane_capacitance: torch.Tensor = 63
    refractory_time: torch.Tensor = 2e-6
    synapse_dac_bias: Union[int, torch.Tensor] = 600
    holdoff_time: torch.Tensor = 0

    log = logger.get("hxtorch.GenericParams")
    logger.set_loglevel(log, logger.LogLevel.TRACE)

    def from_calix_targets(
            self, targets: NeuronCalibTarget,
            neurons: List[halco.LogicalNeuronOnDLS]) -> None:
        """
        Load the params from a calix calibration target.

        :param targets: The calix calibration targets to read the params from.
        :param neurons: The neuron coordinates to which this params object is
            assigned to.
        """
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
        self.i_synin_gm = torch.tensor(targets.i_synin_gm) \
            * torch.ones(2, dtype=torch.int64)

        # synapse_dac_bias
        self.synapse_dac_bias = torch.tensor(targets.synapse_dac_bias)

        # tau_mem, refractory_time, holdoff_time
        # leak, reset, threshold, membrane_capacitance
        for key in ["tau_mem", "refractory_time", "holdoff_time",
                    "leak", "reset", "threshold", "membrane_capacitance"]:
            value = torch.tensor(getattr(targets, key))
            if not value.shape:  # Single number
                setattr(self, key, torch.full(
                    (halco.AtomicNeuronOnDLS.size,), value)[selector])
            if len(value.shape) == 1:  # 1D
                assert_same_values_per_neuron(value)
                setattr(self, key, value[selector])

        # tau_syn
        if (not isinstance(targets.tau_syn, np.ndarray)
                or not targets.tau_syn.shape):
            self.tau_syn = torch.full(
                (halco.AtomicNeuronOnDLS.size,),
                torch.tensor(targets.tau_syn))[selector]
        elif targets.tau_syn.shape == (halco.AtomicNeuronOnDLS.size,):
            assert_same_values_per_neuron(targets.tau_syn)
            self.tau_syn = torch.tensor(
                targets.tau_syn).repeat(2, 1)[:, selector]
        elif targets.tau_syn.shape == (halco.SynapticInputOnNeuron.size,):
            self.tau_syn = torch.tensor(targets.tau_syn).repeat(
                1, halco.AtomicNeuronOnDLS.size)[:, selector]
        elif targets.tau_syn.shape == (halco.SynapticInputOnNeuron.size,
                                       halco.AtomicNeuronOnDLS.size):
            assert_same_values_per_neuron(torch.tensor(targets.tau_syn))
            self.tau_syn = torch.tensor(targets.tau_syn)[:, selector]

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

    def to_calix_targets(
            self, targets: NeuronCalibTarget,
            neurons: List[halco.LogicalNeuronOnDLS]) -> NeuronCalibTarget:
        """
        Add the params to a calix calibration target.

        :param targets: The calix calibration targets to append configuration
            indicated by this params object to.
        :param neurons: The neuron coordinates to which this params object is
            assigned to.

        :return: Returns the calibration target with desired target parameters
            at the neuron coordinates associated with this params object.
        """
        size = len(neurons)
        coords = self._get_an_indices(neurons)
        mapping = self._get_an_to_in_pop_indices(neurons)
        assert len(coords) == len(mapping)

        # i_synin_gm
        i_synin_gm = np.array(self.i_synin_gm)
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
                and self.synapse_dac_bias != targets.synapse_dac_bias):
            raise AttributeError(
                "'synapse_dac_bias' requires same value for all neurons")
        targets.synapse_dac_bias = int(self.synapse_dac_bias)

        # tau_syn
        targets.tau_syn[:, coords] = pq.Quantity(
            self._resize(self.tau_syn, size, rows=2).numpy() * 1e6,
            "us")[:, mapping]

        # e_coba_reversal
        if self.e_coba_reversal is None:
            targets.e_coba_reversal[:, coords] = np.repeat(
                np.array([np.inf, -np.inf])[:, np.newaxis],
                size, axis=1)[:, mapping]
        else:
            targets.e_coba_reversal[:, coords] = self._resize(
                self.e_coba_reversal, size, rows=2).numpy()[:, mapping]

        # e_coba_reference
        if self.e_coba_reference is None:
            targets.e_coba_reference[:, coords] = np.ones((
                halco.SynapticInputOnNeuron.size, size))[:, mapping] * np.nan
        else:
            targets.e_coba_reference[:, coords] = self._resize(
                self.e_coba_reference, size, rows=2).numpy()[:, mapping]

        # tau_mem, refractory_time, holdoff_time
        for key in ["tau_mem", "refractory_time", "holdoff_time"]:
            getattr(targets, key)[coords] = pq.Quantity(
                self._resize(
                    getattr(self, key), size).numpy() * 1e6, "us")[mapping]

        # leak, reset, threshold, membrane_capacitance
        for key in ["leak", "reset", "threshold", "membrane_capacitance"]:
            getattr(targets, key)[coords] = self._resize(
                getattr(self, key), size).numpy()[mapping]
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