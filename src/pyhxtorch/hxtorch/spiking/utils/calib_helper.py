"""
Helpers to handle calibrations
"""
from __future__ import annotations
from typing import Union
import pickle
from pathlib import Path
from dlens_vx_v3 import sta

import _hxtorch_core


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


def calib_from_calix_native(path: Union[str, Path]) -> dict:
    """
    Extract chip config from calix-native pickle dump

    :param path: path to file containing pickled calix result and target.
    """
    with open(path, "rb") as calibfile:
        result = pickle.load(calibfile)
    # return result.to_chip()
    return result


def target_from_calix_native(path: Union[str, Path]) -> dict:
    """
    Extract target dict from calix-native pickle dump

    :param path: path to file containing pickled calix result and target.
    """
    with open(path, "rb") as calibfile:
        result = pickle.load(calibfile)
    return result.target


def nightly_calib_path(name: str = "spiking") -> Path:
    """
    Find path for nightly calibration.
    """
    identifier = _hxtorch_core.get_unique_identifier()
    path = f"/wang/data/calibration/hicann-dls-sr-hx/{identifier}/stable/"\
        f"latest/{name}_cocolist.pbin"
    return Path(path)


def nightly_calix_native_path(name: str = "spiking") -> Path:
    """
    Find path for nightly calibration of calix-native format

    :param name: calibration name prefix.
    """
    identifier = _hxtorch_core.get_unique_identifier()
    path = f"/wang/data/calibration/hicann-dls-sr-hx/{identifier}/stable/"\
        f"latest/{name}_calix-native.pkl"
    return Path(path)
