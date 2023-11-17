"""
Leaky-integrate and fire neurons
"""
from typing import NamedTuple, Tuple, Optional
import torch

from hxtorch.spiking.functional.threshold import threshold
from hxtorch.spiking.functional.unterjubel import Unterjubel


class CUBALIFParams(NamedTuple):

    """ Parameters for CUBA LIF integration and backward path """

    tau_mem_inv: torch.Tensor
    tau_syn_inv: torch.Tensor
    tau_ref: torch.Tensor = torch.tensor(0.)
    v_leak: torch.Tensor = torch.tensor(0.)
    v_th: torch.Tensor = torch.tensor(1.)
    v_reset: torch.Tensor = torch.tensor(0.)
    alpha: float = 50.0
    method: str = "superspike"


# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin, invalid-name, too-many-locals
def cuba_lif_integration(input: torch.Tensor,
                         params: CUBALIFParams,
                         hw_data: Optional[torch.Tensor] = None,
                         dt: float = 1e-6) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Leaky-integrate and fire neuron integration for realization of simple
    spiking neurons with exponential synapses.
    Integrates according to:
        i^{t+1} = i^t * (1 - dt / \tau_{syn}) + x^t
        v^{t+1} = dt / \tau_{men} * (v_l - v^t + i^t) + v^t
        z^{t+1} = 1 if v^{t+1} > params.v_th
        v^{t+1} = params.v_reset if z^{t+1} == 1

    Assumes i^0, v^0 = 0, v_leak
    :note: One `dt` synaptic delay between input and output

    TODO: Issue 3992

    :param input: Input spikes in shape (batch, time, neurons).
    :param params: LIFParams object holding neuron parameters.
    :param dt: Step width of integration.

    :return: Returns the spike trains in shape and membrane trace as a tuple.
        Both tensors are of shape (batch, time, neurons).
    """
    dev = input.device
    T, bs, ps = input.shape
    z, i, v = torch.zeros(bs, ps).to(dev), torch.tensor(0.).to(dev), \
        torch.empty(bs, ps).fill_(params.v_leak).to(dev)
    z_hw, v_cadc, v_madc = None, None, None

    if hw_data is not None:
        z_hw, v_cadc, v_madc = (
            data.to(dev) if data is not None else None for data in hw_data)
        T = min(T, *(
            data.shape[0] for data in (z_hw, v_cadc) if data is not None))
    current, spikes, membrane = [], [], []

    for ts in range(T):
        # Membrane decay
        dv = dt * params.tau_mem_inv * ((params.v_leak - v) + i)
        v = Unterjubel.apply(v + dv, v_cadc[ts]) \
            if v_cadc is not None else v + dv

        # Current
        di = -dt * params.tau_syn_inv * i
        i = i + di + input[ts]

        # Spikes
        spike = threshold(v - params.v_th, params.method, params.alpha)
        z = Unterjubel.apply(spike, z_hw[ts]) if z_hw is not None else spike

        # Reset
        if v_cadc is None:
            v = (1 - z.detach()) * v + z.detach() * params.v_reset

        # Save data
        current.append(i)
        spikes.append(z)
        membrane.append(v)

    return (
        torch.stack(spikes), torch.stack(membrane), torch.stack(current),
        v_madc)
