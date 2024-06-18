"""
Leaky-integrate neurons
"""
from typing import Optional
import torch

from hxtorch.spiking.functional.unterjubel import Unterjubel


# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin, invalid-name, too-many-locals, too-many-arguments
def cuba_li_integration(input: torch.Tensor,
                        *,
                        leak: torch.Tensor,
                        tau_syn: torch.Tensor,
                        tau_mem: torch.Tensor,
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

    :param input: Input graded spike tensor of shape (batch, time, neurons).
    :param leak: The leak voltage as torch.Tensor.
    :param tau_syn: The synaptic time constant as torch.Tensor.
    :param tau_mem: The membrane time constant as torch.Tensor.
    :param hw_data: An optional tuple holding optional hardware observables in
        the order (None, membrane_cadc, membrane_madc).
    :param dt: Integration step width

    :return: Returns the membrane trace in shape (batch, time, neurons).
    """
    dev = input.device
    T, bs, ps = input.shape
    i = torch.tensor(0.).to(dev)
    v = torch.empty(bs, ps, device=dev)
    v[:, :] = leak
    membrane_cadc, membrane_madc = None, None

    if hw_data is not None:
        membrane_cadc, membrane_madc = (
            data.to(dev) if data is not None else None for data in hw_data)
        T = min(membrane_cadc.shape[0], T)
    membrane, current = [], []

    for ts in range(T):
        # Membrane
        dv = dt / tau_mem * (leak - v + i)
        v = Unterjubel.apply(v + dv, membrane_cadc[ts]) \
            if membrane_cadc is not None else v + dv

        # Current
        di = -dt / tau_syn * i
        i = i + di + input[ts]

        # Save data
        membrane.append(v)
        current.append(i)

    return torch.stack(membrane), torch.stack(current), membrane_madc
