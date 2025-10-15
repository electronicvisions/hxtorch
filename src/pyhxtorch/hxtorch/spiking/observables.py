""" Hardware observables object """
from __future__ import annotations

from typing import Optional
from dataclasses import dataclass
import torch

import pyfisch_vx_v3 as fisch
import pygrenade_vx as grenade

import _hxtorch_spiking  # pylint: disable=import-error
from hxtorch.core.observables import HXObservables


SpikeTimes = list[list[list[
    grenade.common.Time
]]]
CADCSamples = list[list[list[
    tuple[
        grenade.common.Time,
        grenade.signal_flow.Int8
    ]
]]]
MADCSamples = list[list[list[
    tuple[
        grenade.common.Time,
        fisch.MADCSampleFromChip.Value
    ]
]]]


@dataclass
class AnalogObservable:
    """
    Dataclass that can hold CADC and MADC data of an analog observable.
    """
    cadc: Optional[torch.Tensor] = None
    madc: Optional[torch.Tensor] = None


@dataclass
class HXTorchObservables(HXObservables):

    def set_data(
        self,
        spikes: SpikeTimes | None = None,
        cadc: CADCSamples | None = None,
        madc: MADCSamples | None = None,
    ) -> None:
        """
        Extract and store hardware observables.

        :param spikes: Raw spike observable data.
        :param cadc: Raw CADC observable data.
        :param madc: Raw MADC observable data.
        """
        if spikes is not None:
            self.spikes = _hxtorch_spiking.extract_spikes(spikes)
        if cadc is not None:
            self.cadc = _hxtorch_spiking.extract_cadc(cadc)
        if madc is not None:
            self.madc = _hxtorch_spiking.extract_madc(madc)
