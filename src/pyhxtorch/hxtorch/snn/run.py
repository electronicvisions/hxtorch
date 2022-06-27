"""
Run function to execute a SNN given in an instance.
"""
from hxtorch.snn.instance import Instance


def run(instance: Instance) -> None:
    """
    Execute the given instance.

    :param instance: The instance representing the computational graph to be
        executed on hardware and/or in software.
    """
    for layer, (input_handle, output_handle) in instance.sorted().items():
        layer.exec_forward(input_handle, output_handle)
