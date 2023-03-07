"""
Run function to execute a SNN given in an experiment.
"""
from typing import Optional
from hxtorch.snn.experiment import Experiment


def run(experiment: Experiment, runtime: Optional[int]) -> None:
    """
    Execute the given experiment.

    TODO: Why is this a standalone function?

    :param experiment: The experiment representing the computational graph to
        be executed on hardware and/or in software.
    :param runtime: Only relevant for hardware experiements. Indicates the
        runtime resolved with experiment.dt.
    """
    if not isinstance(runtime, int) and not experiment.mock:
        raise ValueError("Requested runtime invalid.")

    # Network graph
    data_map = experiment.get_hw_results(runtime)
    for module, inputs, output in experiment.modules.done():
        module.exec_forward(inputs, output, data_map)
