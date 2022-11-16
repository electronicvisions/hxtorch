from _hxtorch import *
import pylogging as logger
from hxtorch import nn

logger.reset()
logger.default_config(level=logger.LogLevel.WARN)
logger.set_loglevel(logger.get("hxtorch"), logger.LogLevel.INFO)
