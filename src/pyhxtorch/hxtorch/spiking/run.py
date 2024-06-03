"""
Run function to execute a SNN given in an experiment.
"""
from typing import Optional
import pylogging as logger
from hxtorch.spiking.experiment import Experiment
from hxtorch.spiking.execution_info import ExecutionInfo

log = logger.get("hxtorch.snn.run")


def run(experiment: Experiment, runtime: Optional[int])\
        -> Optional[ExecutionInfo]:
    """
    Execute the given experiment.

    TODO: Why is this a standalone function?

    :param experiment: The experiment representing the computational graph to
        be executed on hardware and/or in software.
    :param runtime: Only relevant for hardware experiments. Indicates the
        runtime resolved with experiment.dt.
    """
    if not isinstance(runtime, int) and not experiment.mock:
        raise ValueError("Requested runtime invalid.")

    # Network graph
    data_map, execution_info \
        = experiment.get_hw_results(runtime)
    for module, inputs, output in experiment.modules.done():
        module.exec_forward(inputs, output, data_map)

    if execution_info is not None:
        log.TRACE(execution_info.time)

    return execution_info
