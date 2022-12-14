"""
Leaky-integrate and fire neurons
"""
from typing import Callable, NamedTuple, Tuple, Optional
import torch
from hxtorch.snn.functional.superspike import SuperSpike
from hxtorch.snn.functional.unterjubel import Unterjubel


class LIFParams(NamedTuple):

    """ Parameters for LIF integration and backward path """

    tau_mem_inv: torch.Tensor
    tau_syn_inv: torch.Tensor
    tau_ref: torch.Tensor = torch.tensor(0.)
    v_leak: torch.Tensor = torch.tensor(0.)
    v_th: torch.Tensor = torch.tensor(1.)
    v_reset: torch.Tensor = torch.tensor(0.)
    alpha: float = 50.0
    dt: float = torch.tensor(1.0)  # pylint: disable=invalid-name
    activation: Callable = SuperSpike


# Allow redefining builtin for PyTorch consistancy
# pylint: disable=redefined-builtin, invalid-name, too-many-locals
def lif_integration(input: torch.Tensor, params: LIFParams,
                    hw_data: Optional[torch.Tensor] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Leaky-integrate and fire neuron integration for realization of simple
    spiking neurons with exponential synapses.
    Integrates according to:
        i^{t+1} = i^t * (1 - dt / \tau_{syn}) + x^t
        v^{t+1} = dt / \tau_{men} * (v_l - v^t + i^{t+1}) + v^t
        z^{t+1} = 1 if v^{t+1} > params.v_th
        v^{t+1} = params.v_reset if z^{t+1} == 1

    Assumes i^0, v^0 = 0.

    TODO: Issue 3992

    :param input: Input spikes in shape (batch, time, neurons).
    :param params: LIFParams object holding neuron prameters.

    :return: Returns the spike trains in shape and membrane trace as a tuple.
        Both tensors are of shape (batch, time, neurons).
    """
    dev = input.device
    T, bs, ps = input.shape
    z, i, v = torch.zeros(bs, ps).to(dev), torch.tensor(0.).to(dev), \
        torch.empty(bs, ps).fill_(params.v_leak).to(dev)

    if hw_data:
        z_hw = hw_data[0].to(dev)
        v_hw = hw_data[1].to(dev)  # Use CADC values
        spikes, membrane = [z_hw[0]], [v_hw[0]]
        T = min(v_hw.shape[0], T)
    else:
        spikes, membrane = [z], [v]

    for ts in range(T - 1):
        # Current
        i = i * (1 - params.dt * params.tau_syn_inv) + input[ts]

        # Membrane
        dv = params.dt * params.tau_mem_inv * (params.v_leak - v + i)
        v = Unterjubel.apply(dv + v, v_hw[ts + 1]) if hw_data else dv + v

        # Spikes
        spike = params.activation.apply(v - params.v_th, params.alpha)
        z = Unterjubel.apply(spike, z_hw[ts + 1]) if hw_data else spike

        # Reset
        if not hw_data:
            v = (1 - z.detach()) * v + z.detach() * params.v_reset

        # Save data
        spikes.append(z)
        membrane.append(v)

    return torch.stack(spikes), torch.stack(membrane)


class LIF(torch.autograd.Function):

    """ LIF forward mock and backward """

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
