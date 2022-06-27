"""
Leaky-integrate and fire neurons
"""
from typing import Callable, NamedTuple, Tuple
import torch
from hxtorch.snn.functional.superspike import SuperSpike


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
# pylint: disable=redefined-builtin, invalid-name
def lif_integration(input: torch.Tensor, params: LIFParams) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Leaky-integrate and fire neuron integration for realization of simple
    spiking neurons with exponential synapses.
    Integrates according to:
        i^{t+1} = i^t * (1 - dt / \tau_{syn}) + x^t
        v^{t+1} = dt / \tau_{men} * (v_l - v^t) + i^{t+1} + v^t
        z^{t+1} = 1 if v^{t+1} > params.v_th
        v^{t+1} = params.v_reset if z^{t+1} == 1

    Assumes i^0, v^0 = 0.

    TODO: Implement refractory time.

    :param input: Input spikes in shape (batch, time, neurons).
    :param params: LIFParams object holding neuron prameters.

    :return: Returns the spike trains in shape and membrane trace as a tuple.
        Both tensors are of shape (batch, time, neurons).
    """
    dev = input.device
    z, i, v = torch.tensor(0.).to(dev), torch.tensor(0.).to(dev), \
        torch.tensor(0.).to(dev)
    spikes, membrane = list(), list()

    T = input.shape[1]
    for ts in range(T - 1):
        # Current
        i = i * (1 - params.dt * params.tau_syn_inv) + input[:, ts]

        # Membrane
        dv = params.dt * params.tau_mem_inv * (params.v_leak - v) + i
        v = dv + v

        # Spikes
        z = params.activation.apply(v - params.v_th, params.alpha)

        # Reset
        v = (1 - z.detach()) * v + z.detach() * params.v_reset

        # Save data
        spikes.append(z)
        membrane.append(v)

    return torch.stack(spikes).transpose(0, 1), \
        torch.stack(membrane).transpose(0, 1)


class LIF(torch.autograd.Function):

    """ LIF forward mock and backward """

    # Allow redefining builtin for PyTorch consistancy
    # pylint: disable=redefined-builtin, arguments-differ
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        """ Gets overridden """

    # pylint: disable=arguments-differ
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """ Implements LIF backward """
        raise NotImplementedError()
