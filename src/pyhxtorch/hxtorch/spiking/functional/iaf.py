"""
Integrate and fire neurons
"""
# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin, invalid-name, too-many-locals
from typing import NamedTuple, Tuple, Optional
import torch

from hxtorch.spiking.functional.threshold import threshold
from hxtorch.spiking.functional.refractory import refractory_update
from hxtorch.spiking.functional.unterjubel import Unterjubel


class CUBAIAFParams(NamedTuple):
    """ Parameters for IAF integration and backward path """
    tau_mem_inv: torch.Tensor
    tau_syn_inv: torch.Tensor
    tau_ref: torch.Tensor = torch.tensor(0.)
    v_th: torch.Tensor = torch.tensor(1.)
    v_reset: torch.Tensor = torch.tensor(0.)
    alpha: float = 50.0
    method: str = "superspike"


# pylint: disable=too-many-arguments
def iaf_step(z: torch.Tensor, v: torch.Tensor, i: torch.Tensor,
             input: torch.Tensor, z_hw: torch.Tensor, v_hw: torch.Tensor,
             params: NamedTuple, dt: float) -> Tuple[torch.Tensor, ...]:
    """
    Integrate the membrane of a neurons one time step further according to the
    integrate and fire dynamics.
    :param z: The spike tensor at time step t.
    :param v: The membrane tensor at time step t.
    :param i: The current tensor at time step t.
    :param input: The input tensor at time step t (graded spikes).
    :param z_hw: The hardware spikes corresponding to the current time step. In
        case this is None, no HW spikes will be injected.
    :param v_hw: The hardware cadc traces corresponding to the current time
        step. In case this is None, no HW cadc values will be injected.
    :param params: Parameter object holding the LIF parameters.
    :param dt: Integration step width.
    :returns: Returns a tuple (z, v, i) holding the tensors of time step t + 1.
    """
    # Membrane increment
    dv = dt * params.tau_mem_inv * i
    v = Unterjubel.apply(dv + v, v_hw) if z_hw is not None else dv + v
    # Current
    di = -dt * params.tau_syn_inv * i
    i = i + di + input
    # Spikes
    spike = threshold(v - params.v_th, params.method, params.alpha)
    z = Unterjubel.apply(spike, z_hw) if z_hw is not None else spike
    # Reset
    if z_hw is None:
        v = (1 - z.detach()) * v + z.detach() * params.v_reset
    return z, v, i


def cuba_iaf_integration(input: torch.Tensor, params: NamedTuple,
                         hw_data: Optional[torch.Tensor] = None,
                         dt: float = 1e-6) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Leaky-integrate and fire neuron integration for realization of simple
    spiking neurons with exponential synapses.
    Integrates according to:
        v^{t+1} = dt / \tau_{men} * (v_l - v^t + i^t) + v^t
        i^{t+1} = i^t * (1 - dt / \tau_{syn}) + x^t
        z^{t+1} = 1 if v^{t+1} > params.v_th
        v^{t+1} = v_reset if z^{t+1} == 1
    Assumes i^0, v^0 = 0., v_reset
    :note: One `dt` synaptic delay between input and output
    :param input: Input spikes in shape (batch, time, neurons).
    :param params: LIFParams object holding neuron parameters.
    :return: Returns the spike trains in shape and membrane trace as a tuple.
        Both tensors are of shape (batch, time, neurons).
    """
    dev = input.device
    T, bs, ps = input.shape
    z, i, v = torch.zeros(bs, ps).to(dev), torch.tensor(0.).to(dev), \
        torch.empty(bs, ps).fill_(params.v_reset).to(dev)

    if hw_data:
        z_hw = hw_data[0].to(dev)
        v_hw = hw_data[1].to(dev)  # Use CADC values
        T = min(v_hw.shape[0], T)
    spikes, membrane, current = [], [], []

    # Integrate
    for ts in range(T):
        z, v, i = iaf_step(
            z, v, i, input[ts],
            z_hw[ts] if hw_data else None,
            v_hw[ts] if hw_data else None,
            params, dt)

        # Save data
        spikes.append(z)
        membrane.append(v)
        current.append(i)

    return torch.stack(spikes), torch.stack(membrane), torch.stack(current)


def cuba_refractory_iaf_integration(input: torch.Tensor, params: NamedTuple,
                                    hw_data: Optional[torch.Tensor] = None,
                                    dt: float = 1e-6) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate and fire neuron integration for realization of simple
    spiking neurons with exponential synapses and refractory period.
    Integrates according to:
        v^{t+1} = dt / \tau_{men} * i^t + v^t
        i^{t+1} = i^t * (1 - dt / \tau_{syn}) + x^t
        z^{t+1} = 1 if v^{t+1} > params.v_th
        v^{t+1} = params.v_reset if z^{t+1} == 1 or ref^t > 0
        ref^{t+1} -= 1
        ref^{t+1} = params.tau_ref if z^{t+1} == 1
    Assumes i^0, v^0 = 0., v_reset
    :note: One `dt` synaptic delay between input and output
    :param input: Input spikes in shape (batch, time, neurons).
    :param params: LIFParams object holding neuron parameters.
    :return: Returns the spike trains in shape and membrane trace as a tuple.
        Both tensors are of shape (batch, time, neurons).
    """
    dev = input.device
    T, bs, ps = input.shape
    z, i, v = torch.zeros(bs, ps).to(dev), torch.tensor(0.).to(dev), \
        torch.empty(bs, ps).fill_(params.v_reset).to(dev)

    if hw_data:
        z_hw = hw_data[0].to(dev)
        v_hw = hw_data[1].to(dev)  # Use CADC values
        T = min(v_hw.shape[0], T)
    spikes, membrane, current = [], [], []

    # Counter for neurons in refractory period
    ref_state = torch.zeros(ps, dtype=int)

    for ts in range(T):
        z, v, i = iaf_step(
            z, v, i, input[ts],
            z_hw[ts] if hw_data else None,
            v_hw[ts] if hw_data else None,
            params, dt)

        # Refractory update
        z, v, ref_state = refractory_update(z, v, ref_state, params, dt)

        # Save data
        spikes.append(z)
        membrane.append(v)
        current.append(i)

    return torch.stack(spikes), torch.stack(membrane), torch.stack(current)
