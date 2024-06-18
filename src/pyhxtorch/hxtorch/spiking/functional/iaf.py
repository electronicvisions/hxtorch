"""
Integrate and fire neurons
"""
# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin, invalid-name, too-many-locals
from typing import Tuple, Optional
import torch

from hxtorch.spiking.functional.threshold import threshold as spiking_threshold
from hxtorch.spiking.functional.refractory import refractory_update
from hxtorch.spiking.functional.unterjubel import Unterjubel


# pylint: disable=too-many-arguments
def iaf_step(z: torch.Tensor,
             v: torch.Tensor,
             i: torch.Tensor,
             input: torch.Tensor,
             spikes_hw: torch.Tensor,
             membrane_hw: torch.Tensor,
             *,
             reset: torch.Tensor,
             threshold: torch.Tensor,
             tau_syn: torch.Tensor,
             tau_mem: torch.Tensor,
             method: torch.Tensor,
             alpha: torch.Tensor,
             dt: float) -> Tuple[torch.Tensor, ...]:
    """
    Integrate the membrane of a neurons one time step further according to the
    integrate and fire dynamics.

    :param z: The spike tensor at time step t.
    :param v: The membrane tensor at time step t.
    :param i: The current tensor at time step t.
    :param input: The input tensor at time step t (graded spikes).
    :param spikes_hw: The hardware spikes corresponding to the current time
        step. In case this is None, no HW spikes will be injected.
    :param membrane_hw: The hardware CADC traces corresponding to the current
        time step. In case this is None, no HW CADC values will be injected.
    :param reset: The reset voltage as torch.Tensor.
    :param threshold: The threshold voltage as torch.Tensor.
    :param tau_syn: The synaptic time constant as torch.Tensor.
    :param tau_mem: The membrane time constant as torch.Tensor.
    :param method: The method used for the surrogate gradient, e.g.,
        'superspike'.
    :param alpha: The slope of the surrogate gradient in case of 'superspike'.
    :param dt: Integration step width.

    :returns: Returns a tuple (z, v, i) holding the tensors of time step t + 1.
    """
    # Membrane increment
    dv = dt / tau_mem * i
    v = Unterjubel.apply(dv + v, membrane_hw) \
        if membrane_hw is not None else dv + v
    # Current
    di = -dt / tau_syn * i
    i = i + di + input
    # Spikes
    spike = spiking_threshold(v - threshold, method, alpha)
    z = Unterjubel.apply(spike, spikes_hw) if spikes_hw is not None else spike
    # Reset
    if membrane_hw is None:
        v = (1 - z.detach()) * v + z.detach() * reset
    return z, v, i


def cuba_iaf_integration(input: torch.Tensor,
                         *,
                         reset: torch.Tensor,
                         threshold: torch.Tensor,
                         tau_syn: torch.Tensor,
                         tau_mem: torch.Tensor,
                         method: torch.Tensor,
                         alpha: torch.Tensor,
                         hw_data: Optional[torch.Tensor] = None,
                         dt: float = 1e-6) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Leaky-integrate and fire neuron integration for realization of simple
    spiking neurons with exponential synapses.
    Integrates according to:
        v^{t+1} = dt / \tau_{men} * (v_l - v^t + i^t) + v^t
        i^{t+1} = i^t * (1 - dt / \tau_{syn}) + x^t
        z^{t+1} = 1 if v^{t+1} > params.threshold
        v^{t+1} = v_reset if z^{t+1} == 1
    Assumes i^0, v^0 = 0., params.reset
    :note: One `dt` synaptic delay between input and output

    :param input: Input tensor holding 'graded_spikes' in shape (batch, time,
        neurons).
    :param reset: The reset voltage as torch.Tensor.
    :param threshold: The threshold voltage as torch.Tensor.
    :param tau_syn: The synaptic time constant as torch.Tensor.
    :param tau_mem: The membrane time constant as torch.Tensor.
    :param method: The method used for the surrogate gradient, e.g.,
        'superspike'.
    :param alpha: The slope of the surrogate gradient in case of 'superspike'.
    :param hw_data: An optional tuple holding optional hardware observables in
        the order (spikes, membrane_cadc, membrane_madc).
    :param dt: Integration step width.

    :return: Returns tuple of tensors with membrane traces, spikes and synaptic
        current. Tensors are of shape (batch, time, neurons).
    """
    dev = input.device

    T, bs, ps = input.shape
    z, i = torch.zeros(bs, ps).to(dev), torch.tensor(0.).to(dev)
    v = torch.empty(bs, ps, device=dev)
    v[:, :] = reset
    spikes_hw, membrane_cadc, membrane_madc = None, None, None

    if hw_data is not None:
        spikes_hw, membrane_cadc, membrane_madc = (
            data.to(dev) if data is not None else None for data in hw_data)
        T = min(T, *(
            data.shape[0] for data in (
                spikes_hw, membrane_cadc) if data is not None))
    spikes, membrane, current = [], [], []

    # Integrate
    for ts in range(T):
        z, v, i = iaf_step(
            z, v, i, input[ts],
            spikes_hw[ts] if spikes_hw is not None else None,
            membrane_cadc[ts] if membrane_cadc is not None else None,
            reset=reset, threshold=threshold, tau_syn=tau_syn, tau_mem=tau_mem,
            method=method, alpha=alpha, dt=dt)

        # Save data
        spikes.append(z)
        membrane.append(v)
        current.append(i)

    return (torch.stack(spikes), torch.stack(membrane),
            torch.stack(current), membrane_madc)


def cuba_refractory_iaf_integration(input: torch.Tensor,
                                    *,
                                    reset: torch.Tensor,
                                    threshold: torch.Tensor,
                                    tau_syn: torch.Tensor,
                                    tau_mem: torch.Tensor,
                                    refractory_time: torch.Tensor,
                                    method: torch.Tensor,
                                    alpha: torch.Tensor,
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

    :param input: SynapseHandle holding `graded_spikes` in shape (batch, time,
        neurons).
    :param reset: The reset voltage as torch.Tensor.
    :param threshold: The threshold voltage as torch.Tensor.
    :param tau_syn: The synaptic time constant as torch.Tensor.
    :param tau_mem: The membrane time constant as torch.Tensor.
    :param refractory_time: The refractory time constant as torch.Tensor.
    :param method: The method used for the surrogate gradient, e.g.,
        'superspike'.
    :param alpha: The slope of the surrogate gradient in case of 'superspike'.
    :param hw_data: An optional tuple holding optional hardware observables in
        the order (spikes, membrane_cadc, membrane_madc).
    :param dt: Integration step width.

    :return: Returns NeuronHandle holding tensors with membrane traces, spikes
        and synaptic current. Tensors are of shape (batch, time, neurons).
    """
    dev = input.device
    T, bs, ps = input.shape
    z, i, v = torch.zeros(bs, ps).to(dev), torch.tensor(0.).to(dev), \
        torch.empty(bs, ps).fill_(reset).to(dev)
    spikes_hw, membrane_cadc, membrane_madc = None, None, None

    if hw_data is not None:
        spikes_hw, membrane_cadc, membrane_madc = (
            data.to(dev) if data is not None else None for data in hw_data)
        T = min(T, *(
            data.shape[0] for data in (
                spikes_hw, membrane_cadc) if data is not None))
    spikes, membrane, current = [], [], []

    # Counter for neurons in refractory period
    ref_state = torch.zeros(ps, dtype=int)

    for ts in range(T):
        z, v, i = iaf_step(
            z, v, i, input[ts],
            spikes_hw[ts] if spikes_hw is not None else None,
            membrane_cadc[ts] if membrane_cadc is not None else None,
            reset=reset, threshold=threshold, tau_syn=tau_syn, tau_mem=tau_mem,
            method=method, alpha=alpha, dt=dt)

        # Refractory update
        z, v, ref_state = refractory_update(
            z, v, ref_state, spikes_hw[ts] if spikes_hw is not None else None,
            membrane_cadc[ts] if membrane_cadc is not None else None,
            reset=reset, refractory_time=refractory_time, dt=dt)

        # Save data
        spikes.append(z)
        membrane.append(v)
        current.append(i)

    return (torch.stack(spikes), torch.stack(membrane),
            torch.stack(current), membrane_madc)
