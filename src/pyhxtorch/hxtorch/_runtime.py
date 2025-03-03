"""
Shared runtime state for hxtorch.

This module holds the executor and hardware lifecycle functions.
"""
# pylint: disable=global-statement, unnecessary-dunder-call
from typing import Optional
import pygrenade_vx.execution
from _hxtorch_core import (  # pylint: disable=import-error
    _init_hardware_minimal,
    _init_hardware,
    _release_hardware,
    HWDBPath,
    CalibrationPath,
)

_managed_executor = pygrenade_vx.execution.ManagedJITGraphExecutor()
executor: Optional[pygrenade_vx.execution.JITGraphExecutorHandle] = None


def init_hardware_minimal():
    """
    Initialize automatically from the environment without ExperimentInit and
    without any calibration.
    """
    global executor
    executor = _managed_executor.__enter__()
    _init_hardware_minimal(executor)


def init_hardware(
    path: Optional[HWDBPath] = None,
    ann: bool = False,
):
    """
    Initialize the hardware automatically from the environment.

    :param path: Optional path to the hwdb to use.
    :param ann: Boolean flag indicating whether non-spiking or spiking
        calibration is loaded.
    """
    global executor
    executor = _managed_executor.__enter__()
    if isinstance(path, CalibrationPath):
        _init_hardware(
            executor,
            path,
        )
    else:
        _init_hardware(
            executor,
            path,
            ann,
        )


def release_hardware():
    """ Release hardware resource """
    global executor
    executor = None
    _managed_executor.__exit__()
    _release_hardware()
