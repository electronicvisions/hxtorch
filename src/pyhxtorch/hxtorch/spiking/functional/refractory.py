"""
Refractory update for neurons with refractory behaviour
"""
from typing import Tuple, NamedTuple
import torch


# pylint: disable=invalid-name
def refractory_update(z: torch.Tensor, v: torch.Tensor,
                      ref_state: torch.Tensor, params: NamedTuple,
                      dt: float = 1e-6) -> Tuple[torch.Tensor, ...]:
    """
    Update neuron membrane and spikes to account for refractory period.
    This implemention is widly adopted from:
    https://github.com/norse/norse/blob/main/norse/torch/functional/lif_refrac.py
    :param z: The spike tensor at time step t.
    :param v: The membrane tensor at time step t.
    :param ref_state: The refractory state holding the number of time steps the
        neurons has to remain in the refractory period.
    :param params: Parameter object holding the LIF parameters.
    :returns: Returns a tuple (z, v, ref_state) holding the tensors of time
        step t.
    """
    # Refractory mask
    ref_mask = (ref_state > 0).long()
    # Update neuron states
    v = (1 - ref_mask) * v + ref_mask * params.v_reset
    z = (1 - ref_mask) * z
    # Update refractory state
    ref_state = (1 - z) * torch.nn.functional.relu(ref_state - ref_mask) \
        + z * params.tau_ref / dt
    return z, v, ref_state
