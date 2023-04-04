# pylint: disable=unused-import, reimported
import sys as _sys
import pylogging as logger
from _hxtorch_core import *
from hxtorch import perceptron
from hxtorch import spiking
import hxtorch.perceptron as ann
import hxtorch.spiking as snn

_modules = {}
for name, value in _sys.modules.items():
    if "hxtorch.spiking" in name:
        _modules[name.replace("hxtorch.spiking", "hxtorch.snn")] = value
    if "hxtorch.perceptron" in name:
        _modules[name.replace("hxtorch.perceptron", "hxtorch.ann")] = value
_sys.modules.update(_modules)

logger.reset()
logger.default_config(level=logger.LogLevel.WARN)
logger.set_loglevel(logger.get("hxtorch"), logger.LogLevel.INFO)
