"""
Refractory update for neurons with refractory behaviour
"""
from typing import Tuple
import torch

from hxtorch.spiking.functional.unterjubel import Unterjubel


# pylint: disable=invalid-name, too-many-arguments
def refractory_update(
        z: torch.Tensor,
        v: torch.Tensor,
        ref_state: torch.tensor,
        spikes_hw: torch.Tensor,
        membrane_hw: torch.Tensor,
        *,
        reset: torch.Tensor,
        refractory_time: torch.Tensor,
        dt: float) -> Tuple[torch.Tensor, ...]:
    """
    Update neuron membrane and spikes to account for refractory period.
    This implemention is widly adopted from:
    https://github.com/norse/norse/blob/main/norse/torch/functional/lif_refrac.py

    :param z: The spike tensor at time step t.
    :param v: The membrane tensor at time step t.
    :param ref_state: The refractory state holding the number of time steps the
        neurons has to remain in the refractory period.
    :param spikes_hw: The hardware spikes corresponding to the current time
        step. In case this is None, no HW spikes will be injected.
    :param membrnae_hw: The hardware CADC traces corresponding to the current
        time step. In case this is None, no HW CADC values will be injected.
    :param reset: The reset voltage as torch.Tensor.
    :param refractory_time: The refractory time constant as torch.Tensor.
    :param dt: Integration step width.

    :returns: Returns a tuple (z, v, ref_state) holding the tensors of time
        step t.
    """
    # Refractory mask
    ref_mask = (ref_state > 0).to(v.dtype)
    # Update neuron states
    v = (1 - ref_mask) * v + ref_mask * reset
    # Inject HW membrane potential
    v = Unterjubel.apply(v, membrane_hw) if membrane_hw is not None else v
    # Inject HW spike
    z = (1 - ref_mask) * z
    z = Unterjubel.apply(z, spikes_hw) if spikes_hw is not None else z
    # Update refractory state
    ref_state = (1 - z) * torch.nn.functional.relu(ref_state - ref_mask) \
        + z * refractory_time / dt
    return z, v, ref_state
