""" Hardware observables object """
from typing import Optional, Union, Tuple, Dict
from dataclasses import dataclass
import torch

import _hxtorch_spiking
from _hxtorch_spiking import SpikeHandle, CADCHandle, MADCHandle  # pylint: disable=import-error
import pygrenade_vx as grenade


@dataclass
class AnalogObservable:
    """
    Dataclass that can hold CADC and MADC data of an analog observable.
    """
    cadc: Optional[torch.Tensor] = None
    madc: Optional[torch.Tensor] = None


@dataclass
class HardwareObservables:
    """
    Dataclass that holds the observable data measured on hardware, before
    it is fed into post processing.
    """
    spikes: Optional[SpikeHandle] = None
    cadc: Optional[CADCHandle] = None
    madc: Optional[MADCHandle] = None


class HardwareObservablesExtractor:

    _spikes: Dict[
        grenade.network.PopulationOnNetwork, Optional[SpikeHandle]] = None
    _cadc_samples: Dict[
        grenade.network.PopulationOnNetwork, Optional[CADCHandle]] = None
    _madc_samples: Dict[
        grenade.network.PopulationOnNetwork, Optional[MADCHandle]] = None

    def set_data(
            self, network_graph: grenade.network.NetworkGraph,
            result_map: grenade.signal_flow.OutputData) -> None:
        """
        Set the data to be extracted. This method also evokes data extraction.

        :param network_graph: The logical grenade network graph describing the
            logic of th experiment.
        :param result_map: The result map returned by grenade holding all
            recorded hardware observables.
        """
        self._spikes = _hxtorch_spiking.extract_spikes(
            result_map, network_graph)
        self._cadc_samples = _hxtorch_spiking.extract_cadc(
            result_map, network_graph)
        self._madc_samples = _hxtorch_spiking.extract_madc(
            result_map, network_graph)

    def get(
            self, descriptor: Optional[Union[
                grenade.network.PopulationOnNetwork,
                grenade.network.ProjectionOnNetwork,
                Tuple[grenade.network.ProjectionOnNetwork, ...]]] = None) \
            -> HardwareObservables:
        """
        Get the ``HardwareObservables`` assigned to the ``HXModule`` with
        grenade Population/Projection descriptor ``descriptor``.

        :param descriptor: The Population/Projection grenade descriptor to get
            the data for

        :return: The hardware data assigned to the HXModule with descriptor
        ``descriptor``
        """
        return HardwareObservables(
            spikes=self._spikes.get(descriptor),
            cadc=self._cadc_samples.get(descriptor),
            madc=self._madc_samples.get(descriptor))
