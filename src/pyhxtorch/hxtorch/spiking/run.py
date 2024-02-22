"""
Run function to execute a SNN given in an experiment.
"""
from typing import Optional
import pygrenade_vx as grenade
import pylogging as logger
from hxtorch.spiking.experiment import Experiment

log = logger.get("hxtorch.snn.run")


def run(experiment: Experiment, runtime: Optional[int])\
        -> Optional[grenade.signal_flow.ExecutionTimeInfo]:
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
    data_map, execution_time_info = experiment.get_hw_results(runtime)
    for module, inputs, output in experiment.modules.done():
        module.exec_forward(inputs, output, data_map)

    if execution_time_info is not None:
        log.TRACE(execution_time_info)

    return execution_time_info
