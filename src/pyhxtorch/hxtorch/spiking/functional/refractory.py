"""
Refractory update for neurons with refractory behaviour
"""
from typing import Tuple, NamedTuple
import torch

from hxtorch.spiking.functional.unterjubel import Unterjubel


# pylint: disable=invalid-name, too-many-arguments
def refractory_update(z: torch.Tensor, v: torch.Tensor,
                      z_hw: torch.Tensor, v_hw: torch.Tensor,
                      ref_state: torch.Tensor,
                      params: NamedTuple, dt: float = 1e-6) \
        -> Tuple[torch.Tensor, ...]:
    """
    Update neuron membrane and spikes to account for refractory period.
    This implemention is widly adopted from:
    https://github.com/norse/norse/blob/main/norse/torch/functional/lif_refrac.py
    :param z: The spike tensor at time step t.
    :param v: The membrane tensor at time step t.
    :param ref_state: The refractory state holding the number of time steps the
        neurons has to remain in the refractory period.
    :param z_hw: The hardware spikes corresponding to the current time step. In
        case this is None, no HW spikes will be injected.
    :param v_hw: The hardware cadc traces corresponding to the current time
        step. In case this is None, no HW cadc values will be injected.
    :param params: Parameter object holding the LIF parameters.
    :returns: Returns a tuple (z, v, ref_state) holding the tensors of time
        step t.
    """
    # Refractory mask
    ref_mask = (ref_state > 0).to(v.dtype)
    # Update neuron states
    v = (1 - ref_mask) * v + ref_mask * params.reset
    # Inject HW membrane potential
    v = Unterjubel.apply(v, v_hw) if v_hw is not None else v
    # Inject HW spike
    z = (1 - ref_mask) * z
    z = Unterjubel.apply(z, z_hw) if z_hw is not None else z
    # Update refractory state
    ref_state = (1 - z) * torch.nn.functional.relu(ref_state - ref_mask) \
        + z * params.refractory_time / dt
    return z, v, ref_state
