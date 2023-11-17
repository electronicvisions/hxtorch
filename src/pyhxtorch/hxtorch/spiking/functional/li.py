"""
Leaky-integrate neurons
"""
from typing import NamedTuple, Optional
import torch
from hxtorch.spiking.functional.unterjubel import Unterjubel


class CUBALIParams(NamedTuple):

    """ Parameters for CUBA LI integration and backward path """

    tau_mem_inv: torch.Tensor
    tau_syn_inv: torch.Tensor
    v_leak: torch.Tensor = torch.tensor(0.)


# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin, invalid-name, too-many-locals
def cuba_li_integration(input: torch.Tensor, params: CUBALIParams,
                        hw_data: Optional[torch.Tensor] = None,
                        dt: float = 1e-6) -> torch.Tensor:
    """
    Leaky-integrate neuron integration for realization of readout neurons
    with exponential synapses.
    Integrates according to:
        v^{t+1} = dt / \tau_{mem} * (v_l - v^t + i^t) + v^t
        i^{t+1} = i^t * (1 - dt / \tau_{syn}) + x^t

    Assumes i^0, v^0 = 0.
    :note: One `dt` synaptic delay between input and output
    :param input: Input spikes in shape (batch, time, neurons).
    :param params: LIParams object holding neuron parameters.
    :param dt: Integration step width

    :return: Returns the membrane trace in shape (batch, time, neurons).
    """
    dev = input.device
    T, bs, ps = input.shape
    i, v = torch.tensor(0.).to(dev), \
        torch.empty(bs, ps).fill_(params.v_leak).to(dev)
    v_cadc, v_madc = None, None

    if hw_data is not None:
        v_cadc, v_madc = (
            data.to(dev) if data is not None else None for data in hw_data)
        T = min(v_cadc.shape[0], T)
    membrane, current = [], []

    for ts in range(T):
        # Membrane
        dv = dt * params.tau_mem_inv * (params.v_leak - v + i)
        v = Unterjubel.apply(v + dv, v_cadc[ts]) \
            if v_cadc is not None else v + dv

        # Current
        di = -dt * params.tau_syn_inv * i
        i = i + di + input[ts]

        # Save data
        membrane.append(v)
        current.append(i)

    return torch.stack(membrane), torch.stack(current), v_madc
