"""
Run function to execute a SNN given in an experiment.
"""
from typing import Optional
import pylogging as logger
import pygrenade_vx as grenade
from hxtorch.spiking.experiment import Experiment

log = logger.get("hxtorch.snn.run")


def run(experiment: Experiment, runtime: Optional[int])\
        -> Optional:
    """
    Execute the given experiment.

    TODO: Why is this a standalone function?

    :param experiment: The experiment representing the computational graph to
        be executed on hardware and/or in software.
    :param runtime: Only relevant for hardware experiments. Indicates the
        runtime resolved with experiment.dt.
    """
    if not isinstance(runtime, int) and not experiment.mock:
        raise ValueError(
            f"Requested runtime invalid. Expected an int got {type(runtime)}")

    # Network graph
    execution_info = experiment.run(runtime)
    graph_elements = experiment.modules.done()
    for module, inputs, output in graph_elements:
        module.exec_forward(inputs, output)
    # Post processing
    for module, _, output in graph_elements:
        module.post_simulation_processing(output)

    if execution_info is not None:
        # TODO: Generalize to more execution instances
        log.TRACE(
            "Grenade execution health info: ",
            execution_info.execution_instances.get(
                grenade.common.ExecutionInstanceOnExecutor()
            ).execution_health_info
        )
        log.TRACE(
            "Grenade device usage duration: ",
            execution_info.execution_instances.get(
                grenade.common.ExecutionInstanceOnExecutor()
            ).device_usage_duration
        )
        log.TRACE(
            "Grenade realtime duration: ",
            execution_info.execution_instances.get(
                grenade.common.ExecutionInstanceOnExecutor()
            ).realtime_duration
        )

    return execution_info
