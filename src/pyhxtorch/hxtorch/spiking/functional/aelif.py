"""
Adaptive exponential leaky-integrate and fire neurons
"""
from typing import Tuple, Optional, Union
import torch

from hxtorch.spiking.handle import Handle
from hxtorch.spiking.functional.threshold import threshold as spiking_threshold
from hxtorch.spiking.functional.unterjubel import Unterjubel
from hxtorch.spiking.functional.refractory import refractory_update


# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin, invalid-name, too-many-locals
# pylint: disable=too-many-statements, too-many-branches, exec-used
def cuba_aelif_integration(
        input: Union[Tuple[torch.Tensor], torch.Tensor],
        *,
        leak: Union[torch.Tensor, float, int],
        reset: Union[torch.Tensor, float, int],
        threshold: Union[torch.Tensor, float, int],
        tau_syn: Union[torch.Tensor, float, int],
        c_mem: Union[torch.Tensor, float, int],
        g_l: Union[torch.Tensor, float, int],
        refractory_time: Union[torch.Tensor, float, int],
        method: str,
        alpha: float,
        exp_slope: Union[torch.Tensor, float, int],
        exp_threshold: Union[torch.Tensor, float, int],
        subthreshold_adaptation_strength: Union[torch.Tensor, float, int],
        spike_triggered_adaptation_increment: Union[torch.Tensor, float, int],
        tau_adap: Union[torch.Tensor, float, int],
        hw_data: Optional[type(Handle(
            'voltage', 'adaptation', 'spikes'))] = None,
        dt: float = 1e-6,
        leaky: bool = True,
        fire: bool = True,
        refractory: bool = False,
        exponential: bool = False,
        subthreshold_adaptation: bool = False,
        spike_triggered_adaptation: bool = False,
        integration_step_code: str):
    """
    Adaptive exponential leaky-integrate and fire neuron integration for
    realization of AdEx neurons with exponential synapses.
    Certain terms of the differential equations of the membrane voltage v
    and the adaptation current w can be disabled or enabled via flags.

    If all flags are set, it integrates according to:
        i^{t+1} = i^t * (1 - dt / \tau_{syn}) + x^t
        v^{t+1} = dt / c_{mem} * (g_l * (v_l - v^t + T * exp((v^t - v_T) / T))
                                  + i^t - w^t) + v^t
        z^{t+1} = 1 if v^{t+1} > params.threshold
        w^{t+1} = w^t + dt / \tau_{adap} * (a * (v^{t+1} - v_l) - w^t)
                  + b * z^{t+1}
        v^{t+1} = params.reset if z^{t+1} == 1

    Assumes i^0, v^0 = v_leak, if leak term is enabled, else v^0 = 0 and
    w^0 = 0.
    :note: One `dt` synaptic delay between input and output

    :param input: torch.Tensor holding 'graded_spikes' in shape (batch, time,
        neurons) or tuple which holds one of such tensors for each input
        synapse.
    :param leak: The leak voltage.
    :param reset: The reset voltage.
    :param threshold: The threshold voltage.
    :param tau_syn: The synaptic time constant.
    :param c_mem: The membrane capacitance.
    :param g_l: The leak conductance.
    :param refractory_time: The refractory time constant.
    :param method: The method used for the surrogate gradient, e.g.,
        'superspike'.
    :param alpha: The slope of the surrogate gradient in case of 'superspike'.
    :param exp_slope: The exponential slope.
    :param exp_threshold: The exponential threshold.
    :param subthreshold_adaptation_strength: The subthreshold adaptation
        strength.
    :param spike_triggered_adaptation_increment: The spike-triggered adaptation
        increment.
    :param tau_adap: The adaptive time constant.
    :param hw_data: An optional tuple holding optional hardware observables in
        the order (spikes, membrane_cadc, membrane_madc).
    :param dt: Integration step width.
    :param leaky: Flag that enables / disables the leak term when set
        to true / false
    :param fire: Flag that enables / disables firing behaviour when set
        to true / false.
    :param refractory: Flag used to omit the execution of the refractory
        update in case the refractory time is set to zero.
    :param exponential: Flag that enables / disables the exponential term
        in the differential equation for the membrane potential when set
        to true / false.
    :param subthreshold_adaptation: Flag that enables / disables the
        subthreshold adaptation term in the differential equation of the
        adaptation when set to true / false.
    :param spike_triggered_adaptation: Flag that enables / disables
        spike-triggered adaptation when set to true / false.

    :return: Returns tuple holding tensors with spikes, membrane traces,
        adaptation current and synaptic current. Tensors are of shape
        (time, batch, neurons).
    """

    # Merge graded spikes if neuron has inputs from multiple synapses
    if isinstance(input, tuple):
        input = torch.stack(input).sum(0)
    dev = input.device
    T, bs, ps = input.shape
    i = torch.zeros(bs, ps).to(dev)
    z = torch.zeros(bs, ps).to(dev)
    adaptation = torch.zeros(bs, ps).to(dev)
    v = torch.empty(bs, ps, device=dev)
    membrane_cadc_hw = None
    membrane_madc_hw = None
    adaptation_cadc_hw = None
    adaptation_madc_hw = None
    spikes_hw = None

    if hw_data:
        membrane_cadc_hw = hw_data.voltage.cadc.to(dev) \
            if hw_data.voltage.cadc is not None else None
        membrane_madc_hw = hw_data.voltage.madc.to(dev) \
            if hw_data.voltage.madc is not None else None
        adaptation_cadc_hw = hw_data.adaptation.cadc.to(dev) \
            if hw_data.adaptation.cadc is not None else None
        adaptation_madc_hw = hw_data.adaptation.madc.to(dev) \
            if hw_data.adaptation.madc is not None else None
        spikes_hw = hw_data.spikes.to(dev) if hw_data.spikes is not None \
            else None
        T = min(T, *(data.shape[0] for data in (
            membrane_cadc_hw, adaptation_cadc_hw, spikes_hw)
            if data is not None))

    current = torch.empty_like(input, device=dev)
    spikes = torch.empty_like(input, device=dev)
    membrane = torch.empty_like(input, device=dev)
    adaptation_storage = torch.empty_like(input, device=dev)

    # Initialize dictionary to match locals with code variables
    variables = {'i': i, 'z': z, 'adaptation': adaptation,
                 'tau_syn': tau_syn, 'c': c_mem, 'dt': dt,
                 'exp': torch.exp, 'Unterjubel': Unterjubel}

    if leaky:
        v[:] = leak
        variables['leak'] = leak
        if g_l is None:
            g_l = torch.ones(ps)
        variables['g_l'] = g_l
    else:
        v[:, :] = 0.
    if fire:
        variables['spiking_threshold'] = spiking_threshold
        variables['method'] = method
        variables['alpha'] = alpha
        variables['threshold'] = threshold
        variables['z'] = z
        variables['reset'] = reset
    if exponential:
        variables['exp_slope'] = exp_slope
        variables['exp_threshold'] = exp_threshold
        if g_l is None:
            g_l = torch.ones(ps)
        variables['g_l'] = g_l
    adaptation_flag = spike_triggered_adaptation or subthreshold_adaptation
    if adaptation_flag:
        variables['tau_adaptation'] = tau_adap
        variables['adaptation'] = adaptation
    if subthreshold_adaptation:
        v[:] = leak
        variables['leak'] = leak
        variables['a'] = subthreshold_adaptation_strength
    if spike_triggered_adaptation:
        variables['b'] = spike_triggered_adaptation_increment
        variables['z'] = z
    if refractory:
        # Counter for neurons in refractory period
        ref_state = torch.zeros(ps, dtype=int, device=dev)
        variables['ref_state'] = ref_state
        variables['refractory_time'] = refractory_time
        variables['refractory_update'] = refractory_update
        variables['membrane_hw'] = [None] * T
        variables['spikes_hw'] = [None] * T
    if membrane_cadc_hw is not None:
        variables['membrane_hw'] = membrane_cadc_hw
    if adaptation_cadc_hw is not None:
        variables['adaptation_hw'] = adaptation_cadc_hw
    if spikes_hw is not None:
        variables['spikes_hw'] = spikes_hw
    variables['v'] = v
    variables['input'] = input

    # Assert that all tensors are on the right device
    for key, var in variables.items():
        if isinstance(var, torch.Tensor):
            variables[key] = var.to(dev)

    # Integrate
    for ts in range(T):
        variables['ts'] = ts
        # Execute step
        try:
            exec(integration_step_code, variables)
        except Exception as e:
            tb = e.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            lineno = tb.tb_lineno
            raise RuntimeError(
                "An error occured while executing the code of the integration "
                + f"step.\nIn line {lineno} "
                + f"(\"{integration_step_code.splitlines()[lineno-1]}\") in "
                + f"integration_step_code, following error occured:\n{e}") \
                from e
        # Save data
        current[ts, :, :] = variables['i']
        membrane[ts, :, :] = variables['v']
        spikes[ts, :, :] = variables['z']
        adaptation_storage[ts, :, :] = variables['adaptation']

    return (membrane, membrane_madc_hw, current, adaptation_storage,
            adaptation_madc_hw, spikes)
