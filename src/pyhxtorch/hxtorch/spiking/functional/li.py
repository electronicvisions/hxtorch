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


# Allow redefining builtin for PyTorch consistancy
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

    if hw_data:
        v_hw = hw_data[0].to(dev)  # Use CADC values
        T = min(v_hw.shape[0], T)
        # Initialize with first measurement
    membrane, current = [], []

    for ts in range(T):
        # Membrane
        dv = dt * params.tau_mem_inv * (params.v_leak - v + i)
        v = Unterjubel.apply(v + dv, v_hw[ts]) if hw_data else v + dv

        # Current
        di = -dt * params.tau_syn_inv * i
        i = i + di + input[ts]

        # Save data
        membrane.append(v)
        current.append(i)

    return torch.stack(membrane), torch.stack(current)


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
