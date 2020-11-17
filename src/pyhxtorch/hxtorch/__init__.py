from _hxtorch import *
from hxtorch import nn
import pylogging as logger

logger.reset()
logger.default_config(level=logger.LogLevel.WARN)
logger.set_loglevel(logger.get("hxtorch"), logger.LogLevel.INFO)
