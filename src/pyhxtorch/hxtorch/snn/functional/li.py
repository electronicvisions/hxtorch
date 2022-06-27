"""
Leaky-integrate neurons
"""
from typing import NamedTuple
import torch


class LIParams(NamedTuple):

    """ Parameters for LI integration and backward path """

    tau_mem_inv: torch.Tensor
    tau_syn_inv: torch.Tensor
    v_leak: torch.Tensor = torch.tensor(0.)
    dt: float = torch.tensor(1.)  # pylint: disable=invalid-name


# Allow redefining builtin for PyTorch consistancy
# pylint: disable=redefined-builtin, invalid-name
def li_integration(input: torch.Tensor, params: LIParams) -> torch.Tensor:
    """
    Leaky-integrate neuron integration for realization of readout neurons
    with exponential synapses.
    Integrates according to:
        i^{t+1} = i^t * (1 - dt / \tau_{syn}) + x^t
        v^{t+1} = dt / \tau_{mem} * (v_l - v^t) + i^{t+1} + v^t

    Assumes i^0, v^0 = 0.

    :param input: Input spikes in shape (batch, time, neurons).
    :param params: LIParams object holding neuron parameters.

    :return: Returns the membrane trace in shape (batch, time, neurons).
    """
    dev = input.device
    i, v = torch.tensor(0.).to(dev), torch.tensor(0.).to(dev)
    membrane = list()

    T = input.shape[1]
    for ts in range(T - 1):
        # Current
        i = i * (1. - params.dt * params.tau_syn_inv) + input[:, ts]

        # Membrane
        dv = params.dt * params.tau_mem_inv * (params.v_leak - v) + i
        v = dv + v

        # Save data
        membrane.append(v)

    return torch.stack(membrane).transpose(0, 1)
