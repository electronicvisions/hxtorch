"""
Helpers to handle calibrations
"""
from __future__ import annotations
from typing import Union
import pickle
from pathlib import Path
from dlens_vx_v3 import sta

import pygrenade_vx as grenade
import pygrenade_vx.network.abstract as _abstract

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
    return result


def chip_from_calibration_file(path: Union[str, Path]):
    """
    Extract chip config from a known calibration file format.

    Supports portable-binary coco dumps (e.g. ``*.pbin``) and
    calix-native pickle dumps (e.g. ``*_calix-native.pkl``).

    :param path: path to calibration file.
    """
    path = Path(path)
    if path.suffix == ".pkl":
        return calib_from_calix_native(path)
    return chip_from_file(path)


def target_from_calix_native(path: Union[str, Path]) -> dict:
    """
    Extract target dict from calix-native pickle dump

    :param path: path to file containing pickled calix result and target.
    """
    with open(path, "rb") as calibfile:
        result = pickle.load(calibfile)
    return result.target


def fixture_calibration_from_file(path: Union[str, Path]):
    """
    Create a FixtureCalibration from a calibration file path.

    :param path: path to file containing coco dump.
    :return: FixtureCalibration instance.
    """
    chip = chip_from_calibration_file(path)
    return fixture_calibration_from_chip(chip)


def fixture_calibration_from_chip(chip):
    """
    Create a FixtureCalibration from a chip configuration object.

    :param chip: lola chip configuration (e.g. lola.Chip.default_neuron_bypass).
    :return: FixtureCalibration instance.
    """
    # TODO: make this generic
    calibration = _abstract.FixtureCalibration()
    calibration.chips = {
        grenade.common.ExecutionInstanceOnExecutor(
            grenade.common.ExecutionInstanceID(0),
            grenade.common.ConnectionOnExecutor(0)
        ): {
            grenade.common.ChipOnConnection(): chip
        },
    }
    return calibration


def nightly_calib_path(name: str = "spiking") -> Path:
    """
    Find path for nightly calibration.
    """
    identifier = _hxtorch_core.get_unique_identifier()[0]
    path = f"/wang/data/calibration/hicann-dls-sr-hx/{identifier}/stable/"\
        f"latest/{name}_cocolist.pbin"
    return Path(path)


def nightly_calix_native_path(name: str = "spiking") -> Path:
    """
    Find path for nightly calibration of calix-native format

    :param name: calibration name prefix.
    """
    identifier = _hxtorch_core.get_unique_identifier()[0]
    path = f"/wang/data/calibration/hicann-dls-sr-hx/{identifier}/stable/"\
        f"latest/{name}_calix-native.pkl"
    return Path(path)
