from __future__ import annotations
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    import pygrenade_vx as grenade
    from hxtorch.core.experiment import BaseExperiment
    from pyhalco_hicann_dls_vx_v3 import DLSGlobal


class HXBaseModule:
    """
    PyTorch module supplying basic functionality for elements of SNNs that do
    have a representation on hardware
    """

    descriptor = None

    def __init__(
        self,
        experiment: BaseExperiment,
        chip_coordinate: Dict[
            grenade.common.ExecutionInstanceID,
            DLSGlobal
        ] | None = None,
    ) -> None:
        """
        :param experiment: Experiment to append layer to.
        :param chip_coordinate: Chip coordinate to place to.
        """
        self.experiment = experiment
        self.chip_coordinate = chip_coordinate
