"""
Leaky-integrate and fire neurons
"""
from typing import NamedTuple, Tuple, Optional, Union
import dataclasses
import torch

from hxtorch.spiking.calibrated_params import CalibratedParams
from hxtorch.spiking.handle import NeuronHandle, SynapseHandle
from hxtorch.spiking.functional.threshold import threshold as spiking_threshold
from hxtorch.spiking.functional.unterjubel import Unterjubel
from hxtorch.spiking.functional.refractory import refractory_update


class CUBALIFParams(NamedTuple):

    """ Parameters for CUBA LIF integration and backward path """

    tau_mem: torch.Tensor
    tau_syn: torch.Tensor
    refractory_time: torch.Tensor = torch.tensor(0.)
    leak: torch.Tensor = torch.tensor(0.)
    threshold: torch.Tensor = torch.tensor(1.)
    reset: torch.Tensor = torch.tensor(0.)
    alpha: float = 50.0
    method: str = "superspike"


@dataclasses.dataclass(unsafe_hash=True)
class CalibratedCUBALIFParams(CalibratedParams):
    """ Parameters for CUBA LIF integration and backward path """
    alpha: float = 50.0
    method: str = "superspike"


# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments
def cuba_lif_step(
        z: torch.Tensor, v: torch.Tensor, i: torch.Tensor, input: torch.Tensor,
        z_hw: torch.Tensor, v_hw: torch.Tensor,
        params: Union[CalibratedCUBALIFParams, CUBALIFParams],
        dt: float = 1e-6) -> Tuple[torch.Tensor, ...]:
    """
    Integrate the membrane of a neurons one time step further according to the
    Leaky-integrate and fire dynamics.

    :param z: The spike tensor at time step t.
    :param v: The membrane tensor at time step t.
    :param i: The current tensor at time step t.
    :param input: The input tensor at time step t (graded spikes).
    :param z_hw: The hardware spikes corresponding to the current time step. In
        case this is None, no HW spikes will be injected.
    :param v_hw: The hardware cadc traces corresponding to the current time
        step. In case this is None, no HW cadc values will be injected.
    :param params: Parameter object holding the LIF parameters.

    :returns: Returns a tuple (z, v, i) holding the tensors of time step t + 1.
    """
    # Membrane increment
    dv = dt / params.tau_mem * (params.leak - v + i)

    # Current
    i = i * (1 - dt / params.tau_syn) + input

    # Apply integration step
    v = Unterjubel.apply(dv + v, v_hw) if v_hw is not None else dv + v

    # Spikes
    spike = spiking_threshold(
        v - params.threshold, params.method, params.alpha)
    z = Unterjubel.apply(spike, z_hw) if z_hw is not None else spike

    # Reset
    if v_hw is None:
        v = (1 - z.detach()) * v + z.detach() * params.reset

    return z, v, i


# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin, invalid-name, too-many-locals
def cuba_lif_integration(
        input: SynapseHandle,
        params: Union[CalibratedCUBALIFParams, CUBALIFParams],
        hw_data: Optional[torch.Tensor] = None, dt: float = 1e-6) \
        -> Tuple[torch.Tensor, ...]:
    """
    Leaky-integrate and fire neuron integration for realization of simple
    spiking neurons with exponential synapses.
    Integrates according to:
        i^{t+1} = i^t * (1 - dt / \tau_{syn}) + x^t
        v^{t+1} = dt / \tau_{men} * (v_l - v^t + i^t) + v^t
        z^{t+1} = 1 if v^{t+1} > params.threshold
        v^{t+1} = params.reset if z^{t+1} == 1

    Assumes i^0, v^0 = 0, v_leak
    :note: One `dt` synaptic delay between input and output

    TODO: Issue 3992

    :param input: SynapseHandle holding `graded_spikes` in shape (batch, time,
        neurons).
    :param params: LIFParams object holding neuron parameters.
    :param dt: Step width of integration.

    :return: Returns NeuronHandle holding tensors with membrane traces, spikes
        and synaptic current. Tensors are of shape (batch, time, neurons).
    """
    input = input.graded_spikes
    dev = input.device
    T, bs, ps = input.shape
    z, i, v = torch.zeros(bs, ps).to(dev), torch.tensor(0.).to(dev), \
        torch.empty(bs, ps).fill_(params.leak).to(dev)
    spikes_hw, membrane_cadc, membrane_madc = None, None, None

    if hw_data is not None:
        spikes_hw, membrane_cadc, membrane_madc = (
            data.to(dev) if data is not None else None for data in hw_data)
        T = min(T, *(data.shape[0] for data in (
                spikes_hw, membrane_cadc) if data is not None))
    current, spikes, membrane = [], [], []

    # Integrate
    for ts in range(T):
        z, v, i = cuba_lif_step(
            z, v, i, input[ts],
            spikes_hw[ts] if spikes_hw is not None else None,
            membrane_cadc[ts] if membrane_cadc is not None else None,
            params, dt)

        # Save data
        current.append(i)
        spikes.append(z)
        membrane.append(v)

    return NeuronHandle(
        spikes=torch.stack(spikes), membrane_cadc=torch.stack(membrane),
        current=torch.stack(current), membrane_madc=membrane_madc)


# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin, invalid-name, too-many-locals
def cuba_refractory_lif_integration(
        input: SynapseHandle,
        params: Union[CalibratedCUBALIFParams, CUBALIFParams],
        hw_data: Optional[torch.Tensor] = None, dt: float = 1e-6) \
        -> Tuple[torch.Tensor, ...]:
    """
    Leaky-integrate and fire neuron integration for realization of simple
    spiking neurons with exponential synapses and refractory period.

    Integrates according to:
        i^{t+1} = i^t * (1 - dt / \tau_{syn}) + x^t
        v^{t+1} = dt / \tau_{men} * (v_l - v^t + i^{t+1}) + v^t
        z^{t+1} = 1 if v^{t+1} > params.v_th
        v^{t+1} = params.v_reset if z^{t+1} == 1 or ref^{t+1} > 0
        ref^{t+1} = params.tau_ref
        ref^{t+1} -= 1

    Assumes i^0, v^0 = 0.

    :param input: SynapseHandle holding `graded_spikes` in shape (batch, time,
        neurons).
    :param params: LIFParams object holding neuron parameters.

    :return: Returns NeuronHandle holding tensors with membrane traces, spikes
        and synaptic current. Tensors are of shape (batch, time, neurons).
    """
    input = input.graded_spikes
    dev = input.device
    T, bs, ps = input.shape
    z, i, v = torch.zeros(bs, ps).to(dev), torch.tensor(0.).to(dev), \
        torch.empty(bs, ps).fill_(params.leak).to(dev)
    spikes_hw, membrane_cadc, membrane_madc = None, None, None

    if hw_data:
        spikes_hw, membrane_cadc, membrane_madc = (
            data.to(dev) if data is not None else None for data in hw_data)
        T = min(T, *(
            data.shape[0] for data in (
                spikes_hw, membrane_cadc) if data is not None))
    current, spikes, membrane = [], [], []

    # Counter for neurons in refractory period
    ref_state = torch.zeros(ps, dtype=int, device=dev)

    for ts in range(T):
        # Membrane decay
        z, v, i = cuba_lif_step(
            z, v, i, input[ts],
            spikes_hw[ts] if spikes_hw is not None else None,
            membrane_cadc[ts] if membrane_cadc is not None else None,
            params, dt)

        # Refractory update
        z, v, ref_state = refractory_update(
            z, v, spikes_hw[ts] if spikes_hw is not None else None,
            membrane_cadc[ts] if membrane_cadc is not None else None,
            ref_state, params)

        # Save data
        current.append(i)
        spikes.append(z)
        membrane.append(v)

    return NeuronHandle(
        spikes=torch.stack(spikes), membrane_cadc=torch.stack(membrane),
        current=torch.stack(current), membrane_madc=membrane_madc)
