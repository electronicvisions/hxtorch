# pylint: skip-file
"""
Leaky-integrate and fire neurons
"""
from typing import Tuple, Optional
from warnings import warn
import torch

from hxtorch.spiking.functional.threshold import threshold as spiking_threshold
from hxtorch.spiking.functional.unterjubel import Unterjubel
from hxtorch.spiking.functional.refractory import refractory_update
from hxtorch.spiking.handle import Handle


# DEPRECATED
# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments, too-many-locals
def exp_cuba_lif_integration(input: torch.Tensor,
                             *,
                             leak: torch.Tensor,
                             reset: torch.Tensor,
                             threshold: torch.Tensor,
                             tau_syn_exp: torch.Tensor,
                             tau_mem_exp: torch.Tensor,
                             method: torch.Tensor,
                             alpha: torch.Tensor,
                             hw_data: Optional[type(Handle(
                                 'voltage', 'adaptation', 'spikes'))] = None
                             ) -> Tuple[torch.Tensor, ...]:
    warn("The exp_cuba_lif_integration function is deprecated!",
         category=DeprecationWarning)
    """
    Leaky-integrate and fire neuron integration for realization of simple
    spiking neurons with exponential synapses.
    Integrates according to:
    Integrates according to:
        v^{t+1} = tau_mem_exp * v^t + (1 - tau_mem_exp) * i^t
        i^{t+1} = tau_syn_exp * i^t + x^t
        z^{t+1} = 1 if v^{t+1} > params.v_th
        v^{t+1} = params.v_reset if z^{t+1} == 1

    Assumes i^0, v^0 = 0, v_leak
    :note: One `dt` synaptic delay between input and output

    :param input: Tensor holding 'graded_spikes' in shape (batch, time,
        neurons).
    :param leak: The leak voltage as torch.Tensor.
    :param reset: The reset voltage as torch.Tensor.
    :param threshold: The threshold voltage as torch.Tensor.
    :param tau_syn_exp: The synaptic time constant as e^(-dt / \tau_{syn})
        as torch.Tensor.
    :param tau_mem_exp: The membrane time constant as e^(-dt / \tau_{mem})
        as torch.Tensor.
    :param method: The method used for the surrogate gradient, e.g.,
        'superspike'.
    :param alpha: The slope of the surrogate gradient in case of 'superspike'.
    :param hw_data: An optional tuple holding optional hardware observables in
        the order (spikes, membrane_cadc, membrane_madc).
    :param dt: Integration step width.

    :return: Returns tuple holding tensors with membrane traces, spikes
        and synaptic current. Tensors are of shape (batch, time, neurons).
    """
    dev = input.device
    T, bs, ps = input.shape
    z = torch.zeros(bs, ps).to(dev)
    v = torch.empty(bs, ps, device=dev)
    v[:, :] = leak
    i = torch.zeros(bs, ps).to(dev)

    current, spikes, membrane = [], [], []

    spikes_hw, membrane_cadc, membrane_madc = None, None, None
    if hw_data is not None:
        spikes_hw = hw_data.spikes.to(dev) if hw_data.spikes is not None else \
            None
        membrane_cadc = hw_data.voltage.cadc.to(dev) if hw_data.voltage.cadc \
            is not None else None
        membrane_madc = hw_data.voltage.madc.to(dev) if hw_data.voltage.madc \
            is not None else None

        T = min(T, *(data.shape[0] for data in (
            spikes_hw, membrane_cadc) if data is not None))

    for ts in range(T):
        # compute voltage updates
        v = tau_mem_exp * (v - leak) + leak + (1 - tau_mem_exp) * i
        v = Unterjubel.apply(v, membrane_cadc[ts]) \
            if membrane_cadc is not None else v

        # compute current updates
        i = tau_syn_exp * i + input[ts]

        # compute spikes
        out_spikes = spiking_threshold(v - threshold, method, alpha)
        z = Unterjubel.apply(out_spikes, spikes_hw[ts]) \
            if spikes_hw is not None else out_spikes

        # reset
        if membrane_cadc is None:
            v = (1 - z.detach()) * v + z.detach() * reset

        membrane.append(v)
        spikes.append(z)
        current.append(i)

    spikes = torch.stack(spikes)
    mem = torch.stack(membrane)
    syn = torch.stack(current)

    return (spikes, mem, syn, membrane_madc)
