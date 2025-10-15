""" Hardware observables object """
from typing import NewType, Any
from dataclasses import dataclass

HXObservableHandle = NewType("HXObservableHandle", Any)


@dataclass
class HXObservables:
    spikes: HXObservableHandle = None
    cadc: HXObservableHandle = None
    madc: HXObservableHandle = None

    def set_data(
        self,
        spikes: HXObservableHandle = None,
        cadc: HXObservableHandle = None,
        madc: HXObservableHandle = None,
    ) -> None:
        """
        Set hardware observable data.

        :param spikes: Spike observables handle.
        :param cadc: CADC observables handle.
        :param madc: MADC observables handle.
        """
        self.spikes = spikes
        self.cadc = cadc
        self.madc = madc
