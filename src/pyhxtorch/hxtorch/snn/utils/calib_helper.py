"""
Helpers to handle calibrations
"""
from pathlib import Path
from dlens_vx_v3 import sta

import _hxtorch


def chip_from_portable_binary(data: bytes) -> dict:
    """
    Convert portable binary data to chip object.

    :param data: Coco list in portable binary format.
    :return: lola chip configuration.
    """
    dumper = sta.DumperDone()
    sta.from_portablebinary(dumper, data)
    return sta.convert_to_chip(dumper)


def chip_from_file(path: str) -> dict:
    """
    Extract chip config from coco file dump

    :param path: path to file containing coco dump.
    """
    with open(path, 'rb') as fd:
        data = fd.read()
    return chip_from_portable_binary(data)


def nightly_calib_path(name: str = "spiking") -> Path:
    """
    Find path for nightly calibration.
    """
    identifier = _hxtorch.get_unique_identifier()
    path = f"/wang/data/calibration/hicann-dls-sr-hx/{identifier}/stable/"\
        f"latest/{name}_cocolist.pbin"
    return Path(path)
