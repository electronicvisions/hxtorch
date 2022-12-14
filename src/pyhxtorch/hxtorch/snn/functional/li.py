"""
Leaky-integrate neurons
"""
from typing import NamedTuple, Optional
import torch
from hxtorch.snn.functional.unterjubel import Unterjubel


class LIParams(NamedTuple):

    """ Parameters for LI integration and backward path """

    tau_mem_inv: torch.Tensor
    tau_syn_inv: torch.Tensor
    v_leak: torch.Tensor = torch.tensor(0.)
    dt: float = torch.tensor(1e-6)  # pylint: disable=invalid-name


# Allow redefining builtin for PyTorch consistancy
# pylint: disable=redefined-builtin, invalid-name
def li_integration(input: torch.Tensor, params: LIParams,
                   hw_data: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Leaky-integrate neuron integration for realization of readout neurons
    with exponential synapses.
    Integrates according to:
        i^{t+1} = i^t * (1 - dt / \tau_{syn}) + x^t
        v^{t+1} = dt / \tau_{mem} * (v_l - v^t + i^{t+1}) + v^t

    Assumes i^0, v^0 = 0.

    :param input: Input spikes in shape (batch, time, neurons).
    :param params: LIParams object holding neuron parameters.

    :return: Returns the membrane trace in shape (batch, time, neurons).
    """
    dev = input.device
    T, bs, ps = input.shape
    i, v = torch.tensor(0.).to(dev), \
        torch.empty(bs, ps).fill_(params.v_leak).to(dev)

    if hw_data:
        v_hw = hw_data[0].to(dev)  # Use CADC values
        T = min(v_hw.shape[1], T)
        # Initialize with first measurement
        membrane = [v_hw[0]]
    else:
        membrane = [v]

    for ts in range(T - 1):
        # Current
        i = i * (1. - params.dt * params.tau_syn_inv) + input[ts]

        # Membrane
        dv = params.dt * params.tau_mem_inv * (params.v_leak - v + i)
        v = Unterjubel.apply(dv + v, v_hw[ts + 1]) if hw_data else dv + v

        # Save data
        membrane.append(v)

    return torch.stack(membrane)


class LI(torch.autograd.Function):

    """ LI forward mock and backward """

    # Allow redefining builtin for PyTorch consistency
    # pylint: disable=redefined-builtin, arguments-differ
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        """ Gets overridden """

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """ Implements LIF backward """
        raise NotImplementedError()
