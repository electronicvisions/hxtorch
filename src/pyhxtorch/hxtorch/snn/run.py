"""
Run function to execute a SNN given in an instance.
"""
from typing import Optional
from hxtorch.snn.instance import Instance


def run(instance: Instance, runtime: Optional[int]) -> None:
    """
    Execute the given instance.

    TODO: Why is this a standalone function?

    :param instance: The instance representing the computational graph to be
        executed on hardware and/or in software.
    :param runtime: Only relevant for hardware experiements. Indicates the
        runtime resolved with instance.dt.
    """
    if not isinstance(runtime, int) and not instance.mock:
        raise ValueError("Requested runtime invalid.")

    # Network graph
    data_map = instance.get_hw_results(runtime)
    for module, inputs, output in instance.modules.done():
        module.exec_forward(inputs, output, data_map)
