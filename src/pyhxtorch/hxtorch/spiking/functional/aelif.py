"""
Adaptive exponential leaky-integrate and fire neurons
"""
from functools import partial
from typing import Tuple, Optional, Union, Callable
import torch

from hxtorch.spiking.handle import Handle
from hxtorch.spiking.functional.unterjubel import Unterjubel
from hxtorch.spiking.functional.refractory import refractory_update
from hxtorch.spiking.functional.mock import (
    RandomNoise, RandomNoiseAdd, Bounds, saturate, FixedPatternNoiseMultiply,
    FixedPatternNoiseExponential)
from hxtorch.spiking.functional.surrogates import superspike
from hxtorch.core.modules.population import ModuleParameterType
from hxtorch.core.parameter import ParameterType, MockParameter


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
        spike_surrogate: Callable = partial(superspike, alpha=50),
        exp_slope: Union[torch.Tensor, float, int],
        exp_threshold: Union[torch.Tensor, float, int],
        subthreshold_adaptation_strength: Union[torch.Tensor, float, int],
        spike_triggered_adaptation_increment: Union[torch.Tensor, float, int],
        tau_adap: Union[torch.Tensor, float, int],
        hw_data: Optional[type(Handle('voltage', 'current_exc', 'current_inh',
                                      'adaptation', 'spikes'))] = None,
        dt: float = 1e-6,
        trace_noise_current: Optional[RandomNoise] = None,
        trace_noise_voltage: Optional[RandomNoise] = None,
        trace_noise_adaptation: Optional[RandomNoise] = None,
        cadc_readout_noise_current: Optional[RandomNoise] = None,
        cadc_readout_noise_voltage: Optional[RandomNoise] = None,
        cadc_readout_noise_adaptation: Optional[RandomNoise] = None,
        cadc_readout_bounds_current: Optional[Bounds] = None,
        cadc_readout_bounds_voltage: Optional[Bounds] = None,
        cadc_readout_bounds_adaptation: Optional[Bounds] = None,
        dynamic_range_current: Optional[Bounds] = None,
        dynamic_range_voltage: Optional[Bounds] = None,
        dynamic_range_adaptation: Optional[Bounds] = None,
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
    :param spike_surrogate: Surrogate function for the spike triggering
        mechanism.
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
    :param trace_noise_current: `RandomNoise` object which generates random
        noise that is added onto the simulation result of the synaptic input
        current in each time step during simulation in order to mock
        temporal noise. If set to `None`, no temporal noise will be applied to
        the current trace.
    :param trace_noise_voltage: `RandomNoise` object which generates random
        noise that is added onto the simulation result of the membrane
        voltage increment in each time step during simulation in order to mock
        temporal noise. If set to `None`, no temporal noise will be applied to
        the voltage trace.
    :param trace_noise_adaptation: `RandomNoise` object which generates random
        noise that is added onto the simulation result of the adaptation in
        each time step during simulation in order to mock temporal noise. If
        set to `None`, no temporal noise will be applied to the adaptation
        trace.
    :param cadc_readout_noise_current: `RandomNoise` object which generates
        random noise that is added onto the simulation result of the
        synaptic current once after the simulation in order to mock readout
        noise.  If set to `None`, no readout noise will be applied to the
        synaptic current.
    :param cadc_readout_noise_voltage: `RandomNoise` object which generates
        random noise that is added onto the simulation result of the
        membrane voltage once after the simulation in order to mock readout
        noise.  If set to `None`, no readout noise will be applied to the
        membrane voltage.
    :param cadc_readout_noise_adaptation: `RandomNoise` object which
        generates random noise that is added onto the simulation result of
        the adaptation once after the simulation in order to mock readout
        noise.  If set to `None`, no readout noise will be applied to the
        adaptation.
    :param cadc_readout_bounds_current: `Bounds` object that specifies lower
        and upper bounds of the value range the resulting observable data for
        current is clamped to after the simulation is finished. If set to
        `None`, no clamping is performed on the current data after the
        simulation.
    :param cadc_readout_bounds_voltage: `Bounds` object that specifies lower
        and upper bounds of the value range the resulting observable data for
        voltage is clamped to after the simulation is finished. If set to
        `None`, no clamping is performed on the voltage data after the
        simulation.
    :param cadc_readout_bounds_adaptation: `Bounds` object that specifies
        lower and upper bounds of the value range the resulting observable data
        for adaptation is clamped to after the simulation is finished. If set
        to `None`, no clamping is performed on the adaptation data after the
        simulation.
    :param dynamic_range_current: `Bounds` object that specifies lower and
        upper bounds of the value range for the current trace. If a bound is
        exceeded in a time step in simulation, the value for the current is
        clamped to the according bound. If set to `None`, no clamping is
        performed on the current trace throughout simulation.
    :param dynamic_range_voltage: `Bounds` object that specifies lower and
        upper bounds of the value range for the voltage trace. If a bound is
        exceeded in a time step in simulation, the value for the voltage is
        clamped to the according bound. If set to `None`, no clamping is
        performed on the voltage trace throughout simulation.
    :param dynamic_range_adaptation: `Bounds` object that specifies lower and
        upper bounds of the value range for the adaptation trace. If a bound is
        exceeded in a time step in simulation, the value for the adaptation is
        clamped to the according bound. If set to `None`, no clamping is
        performed on the adaptation trace throughout simulation.
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
    current_exc_cadc_hw = None
    current_exc_madc_hw = None
    current_inh_cadc_hw = None
    current_inh_madc_hw = None
    adaptation_cadc_hw = None
    adaptation_madc_hw = None
    spikes_hw = None

    if hw_data:
        membrane_cadc_hw = hw_data.voltage.cadc.to(dev) \
            if hw_data.voltage.cadc is not None else None
        membrane_madc_hw = hw_data.voltage.madc.to(dev) \
            if hw_data.voltage.madc is not None else None
        current_exc_cadc_hw = hw_data.current_exc.cadc.to(dev) \
            if hw_data.current_exc.cadc is not None else None
        current_exc_madc_hw = hw_data.current_exc.madc.to(dev) \
            if hw_data.current_exc.madc is not None else None
        current_inh_cadc_hw = hw_data.current_inh.cadc.to(dev) \
            if hw_data.current_inh.cadc is not None else None
        current_inh_madc_hw = hw_data.current_inh.madc.to(dev) \
            if hw_data.current_inh.madc is not None else None
        adaptation_cadc_hw = hw_data.adaptation.cadc.to(dev) \
            if hw_data.adaptation.cadc is not None else None
        adaptation_madc_hw = hw_data.adaptation.madc.to(dev) \
            if hw_data.adaptation.madc is not None else None
        spikes_hw = hw_data.spikes.to(dev) if hw_data.spikes is not None \
            else None
        T = min((T, *(data.shape[0] for data in (
            membrane_cadc_hw, adaptation_cadc_hw, spikes_hw)
            if data is not None)))

    current = torch.empty_like(input, device=dev)
    spikes = torch.empty_like(input, device=dev)
    membrane = torch.empty_like(input, device=dev)
    adaptation_storage = torch.empty_like(input, device=dev)

    # Set up mock attributes
    if trace_noise_current is None:
        trace_noise_current = RandomNoise(None, dev)
    if trace_noise_voltage is None:
        trace_noise_voltage = RandomNoise(None, dev)
    if trace_noise_adaptation is None:
        trace_noise_adaptation = RandomNoise(None, dev)
    trace_noise_current.to(dev)
    trace_noise_voltage.to(dev)
    trace_noise_adaptation.to(dev)

    lower_limit = -torch.inf
    upper_limit = torch.inf
    if dynamic_range_current is None:
        dynamic_range_current = Bounds(lower_limit, upper_limit)
    if dynamic_range_voltage is None:
        dynamic_range_voltage = Bounds(lower_limit, upper_limit)
    if dynamic_range_adaptation is None:
        dynamic_range_adaptation = Bounds(lower_limit, upper_limit)
    dynamic_range_current.to(dev)
    dynamic_range_voltage.to(dev)
    dynamic_range_adaptation.to(dev)

    # Initialize dictionary to match locals with code variables
    variables = {'i': i, 'z': z, 'adaptation': adaptation,
                 'tau_syn': tau_syn, 'c': c_mem, 'dt': dt,
                 'exp': torch.exp, 'Unterjubel': Unterjubel,
                 'random_noise_current': trace_noise_current,
                 'random_noise_voltage': trace_noise_voltage,
                 'random_noise_adaptation': trace_noise_adaptation,
                 'RandomNoiseAdd': RandomNoiseAdd,
                 'dynamic_range_current': dynamic_range_current,
                 'dynamic_range_voltage': dynamic_range_voltage,
                 'dynamic_range_adaptation': dynamic_range_adaptation,
                 'saturate': saturate}

    if leaky:
        v[:] = leak
        variables['leak'] = leak
        if g_l is None:
            g_l = torch.ones(ps)
        variables['g_l'] = g_l
    else:
        v[:, :] = 0.
    if fire:
        variables['spike_surrogate'] = spike_surrogate
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

    # Add CADC readout noise
    if cadc_readout_noise_current is not None:
        current = RandomNoiseAdd.apply(
            current, cadc_readout_noise_current.sample(current.shape))
    if cadc_readout_noise_voltage is not None and membrane_cadc_hw is None:
        membrane = RandomNoiseAdd.apply(
            membrane, cadc_readout_noise_voltage.sample(membrane.shape))
    if (cadc_readout_noise_adaptation is not None
            and adaptation_cadc_hw is None):
        adaptation_storage = RandomNoiseAdd.apply(
            adaptation_storage,
            cadc_readout_noise_adaptation.sample(adaptation_storage.shape))

    # Clamp results at CADC readout bounds
    if cadc_readout_bounds_current is not None:
        membrane = saturate(membrane, cadc_readout_bounds_current)
    if cadc_readout_bounds_voltage is not None:
        membrane = saturate(membrane, cadc_readout_bounds_voltage)
    if cadc_readout_bounds_adaptation is not None:
        adaptation_storage = saturate(
            adaptation_storage, cadc_readout_bounds_adaptation)

    return (membrane, membrane_madc_hw, current, current_exc_cadc_hw,
            current_exc_madc_hw, current_inh_cadc_hw, current_inh_madc_hw,
            adaptation_storage, adaptation_madc_hw, spikes)


# Allow redefining builtin for PyTorch consistency
# pylint: disable=redefined-builtin, invalid-name, too-many-locals
# pylint: disable=too-many-statements, too-many-branches, exec-used
def cuba_aelif_integration_fpn(
        input: Union[Tuple[torch.Tensor], torch.Tensor],
        *,
        leak: ModuleParameterType,
        reset: ModuleParameterType,
        threshold: ModuleParameterType,
        tau_syn: ModuleParameterType,
        c_mem: ModuleParameterType,
        g_l: ModuleParameterType,
        refractory_time: Union[torch.Tensor, float, int],
        spike_surrogate: Callable = partial(superspike, alpha=50),
        exp_slope: ModuleParameterType,
        exp_threshold: ModuleParameterType,
        subthreshold_adaptation_strength: ModuleParameterType,
        spike_triggered_adaptation_increment: ModuleParameterType,
        tau_adap: ModuleParameterType,
        hw_data: Optional[type(Handle('voltage', 'current_exc', 'current_inh',
                                      'adaptation', 'spikes'))] = None,
        dt: float = 1e-6,
        trace_noise_current: Optional[RandomNoise] = None,
        trace_noise_voltage: Optional[RandomNoise] = None,
        trace_noise_adaptation: Optional[RandomNoise] = None,
        cadc_readout_noise_current: Optional[RandomNoise] = None,
        cadc_readout_noise_voltage: Optional[RandomNoise] = None,
        cadc_readout_noise_adaptation: Optional[RandomNoise] = None,
        cadc_readout_bounds_current: Optional[Bounds] = None,
        cadc_readout_bounds_voltage: Optional[Bounds] = None,
        cadc_readout_bounds_adaptation: Optional[Bounds] = None,
        dynamic_range_current: Optional[Bounds] = None,
        dynamic_range_voltage: Optional[Bounds] = None,
        dynamic_range_adaptation: Optional[Bounds] = None,
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
    :param spike_surrogate: Surrogate function for the spike triggering
        mechanism.
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
    :param trace_noise_current: `RandomNoise` object which generates random
        noise that is added onto the simulation result of the synaptic input
        current in each time step during simulation in order to mock
        temporal noise. If set to `None`, no temporal noise will be applied to
        the current trace.
    :param trace_noise_voltage: `RandomNoise` object which generates random
        noise that is added onto the simulation result of the membrane
        voltage increment in each time step during simulation in order to mock
        temporal noise. If set to `None`, no temporal noise will be applied to
        the voltage trace.
    :param trace_noise_adaptation: `RandomNoise` object which generates random
        noise that is added onto the simulation result of the adaptation in
        each time step during simulation in order to mock temporal noise. If
        set to `None`, no temporal noise will be applied to the adaptation
        trace.
    :param cadc_readout_noise_current: `RandomNoise` object which generates
        random noise that is added onto the simulation result of the
        synaptic current once after the simulation in order to mock readout
        noise.  If set to `None`, no readout noise will be applied to the
        synaptic current.
    :param cadc_readout_noise_voltage: `RandomNoise` object which generates
        random noise that is added onto the simulation result of the
        membrane voltage once after the simulation in order to mock readout
        noise.  If set to `None`, no readout noise will be applied to the
        membrane voltage.
    :param cadc_readout_noise_adaptation: `RandomNoise` object which
        generates random noise that is added onto the simulation result of
        the adaptation once after the simulation in order to mock readout
        noise.  If set to `None`, no readout noise will be applied to the
        adaptation.
    :param cadc_readout_bounds_current: `Bounds` object that specifies lower
        and upper bounds of the value range the resulting observable data for
        current is clamped to after the simulation is finished. If set to
        `None`, no clamping is performed on the current data after the
        simulation.
    :param cadc_readout_bounds_voltage: `Bounds` object that specifies lower
        and upper bounds of the value range the resulting observable data for
        voltage is clamped to after the simulation is finished. If set to
        `None`, no clamping is performed on the voltage data after the
        simulation.
    :param cadc_readout_bounds_adaptation: `Bounds` object that specifies
        lower and upper bounds of the value range the resulting observable data
        for adaptation is clamped to after the simulation is finished. If set
        to `None`, no clamping is performed on the adaptation data after the
        simulation.
    :param dynamic_range_current: `Bounds` object that specifies lower and
        upper bounds of the value range for the current trace. If a bound is
        exceeded in a time step in simulation, the value for the current is
        clamped to the according bound. If set to `None`, no clamping is
        performed on the current trace throughout simulation.
    :param dynamic_range_voltage: `Bounds` object that specifies lower and
        upper bounds of the value range for the voltage trace. If a bound is
        exceeded in a time step in simulation, the value for the voltage is
        clamped to the according bound. If set to `None`, no clamping is
        performed on the voltage trace throughout simulation.
    :param dynamic_range_adaptation: `Bounds` object that specifies lower and
        upper bounds of the value range for the adaptation trace. If a bound is
        exceeded in a time step in simulation, the value for the adaptation is
        clamped to the according bound. If set to `None`, no clamping is
        performed on the adaptation trace throughout simulation.
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
    current_exc_cadc_hw = None
    current_exc_madc_hw = None
    current_inh_cadc_hw = None
    current_inh_madc_hw = None
    adaptation_cadc_hw = None
    adaptation_madc_hw = None
    spikes_hw = None

    if hw_data:
        membrane_cadc_hw = hw_data.voltage.cadc.to(dev) \
            if hw_data.voltage.cadc is not None else None
        membrane_madc_hw = hw_data.voltage.madc.to(dev) \
            if hw_data.voltage.madc is not None else None
        current_exc_cadc_hw = hw_data.current_exc.cadc.to(dev) \
            if hw_data.current_exc.cadc is not None else None
        current_exc_madc_hw = hw_data.current_exc.madc.to(dev) \
            if hw_data.current_exc.madc is not None else None
        current_inh_cadc_hw = hw_data.current_inh.cadc.to(dev) \
            if hw_data.current_inh.cadc is not None else None
        current_inh_madc_hw = hw_data.current_inh.madc.to(dev) \
            if hw_data.current_inh.madc is not None else None
        adaptation_cadc_hw = hw_data.adaptation.cadc.to(dev) \
            if hw_data.adaptation.cadc is not None else None
        adaptation_madc_hw = hw_data.adaptation.madc.to(dev) \
            if hw_data.adaptation.madc is not None else None
        spikes_hw = hw_data.spikes.to(dev) if hw_data.spikes is not None \
            else None
        T = min((T, *(data.shape[0] for data in (
            membrane_cadc_hw, adaptation_cadc_hw, spikes_hw)
            if data is not None)))

    for param in (leak, reset, threshold, tau_syn, c_mem, g_l, refractory_time,
                  exp_threshold, exp_slope, subthreshold_adaptation_strength,
                  spike_triggered_adaptation_increment, tau_adap):
        if isinstance(param, torch.Tensor):
            param = param.to(dev)
        elif (isinstance(param, ParameterType)
                and isinstance(param.model_value, torch.Tensor)):
            param.model_value = param.model_value.to(dev)
        if (isinstance(param, MockParameter)
                and isinstance(param.mean, torch.Tensor)):
            param.mean = param.mean.to(dev)

    # Extract mean values
    threshold_mean = threshold.mean if isinstance(threshold, MockParameter) \
        else None
    tau_syn_mean = tau_syn.mean if isinstance(tau_syn, MockParameter) else None
    c_mem_mean = c_mem.mean if isinstance(c_mem, MockParameter) else None
    g_l_mean = g_l.mean if isinstance(g_l, MockParameter) else None
    exp_threshold_mean = exp_threshold.mean \
        if isinstance(exp_threshold, MockParameter) else None
    exp_slope_mean = exp_slope.mean if isinstance(exp_slope, MockParameter) \
        else None
    subthreshold_adaptation_strength_mean = (
        subthreshold_adaptation_strength.mean
        if isinstance(subthreshold_adaptation_strength, MockParameter)
        else None)
    spike_triggered_adaptation_increment_mean = (
        spike_triggered_adaptation_increment.mean
        if isinstance(spike_triggered_adaptation_increment, MockParameter)
        else None)
    tau_adap_mean = tau_adap.mean if isinstance(tau_adap, MockParameter) \
        else None

    # Convert to numbers
    leak = leak.model_value if isinstance(leak, ParameterType) else leak
    g_l = g_l.model_value if isinstance(g_l, ParameterType) else g_l
    g_l = torch.ones(ps) if g_l is None else g_l
    threshold = threshold.model_value if isinstance(threshold, ParameterType) \
        else threshold
    reset = reset.model_value if isinstance(reset, ParameterType) else reset
    tau_syn = tau_syn.model_value if isinstance(tau_syn, ParameterType) \
        else tau_syn
    c_mem = c_mem.model_value if isinstance(c_mem, ParameterType) else c_mem
    exp_slope = exp_slope.model_value if isinstance(exp_slope, ParameterType) \
        else exp_slope
    exp_threshold = exp_threshold.model_value \
        if isinstance(exp_threshold, ParameterType) else exp_threshold
    subthreshold_adaptation_strength = (
        subthreshold_adaptation_strength.model_value
        if isinstance(subthreshold_adaptation_strength, ParameterType)
        else subthreshold_adaptation_strength)
    spike_triggered_adaptation_increment = (
        spike_triggered_adaptation_increment.model_value
        if isinstance(spike_triggered_adaptation_increment, ParameterType)
        else spike_triggered_adaptation_increment)
    tau_adap = tau_adap.model_value if isinstance(tau_adap, ParameterType) \
        else tau_adap

    # Calculate one-value factors
    fi1 = 1 - dt / tau_syn
    fi1_mean = None if tau_syn_mean is None else 1 - dt / tau_syn_mean
    fv1 = dt / c_mem
    fv1_mean = None if c_mem_mean is None else dt / c_mem_mean
    fv2 = g_l * exp_slope
    fv2_mean = None
    if g_l_mean is None and exp_slope_mean is not None:
        fv2_mean = g_l * exp_slope_mean
    if g_l_mean is not None and exp_slope_mean is None:
        fv2_mean = g_l_mean * exp_slope
    if g_l_mean is not None and exp_slope_mean is not None:
        fv2_mean = g_l_mean * exp_slope_mean
    fv3 = 1 / exp_slope
    fv3_mean = None if exp_slope_mean is None else 1 / exp_slope_mean
    fa1 = 1 - dt / tau_adap
    fa1_mean = None if tau_adap_mean is None else 1 - dt / tau_adap_mean
    fa2 = subthreshold_adaptation_strength * dt / tau_adap
    fa2_mean = None
    if (subthreshold_adaptation_strength_mean is None
            and tau_adap_mean is not None):
        fa2_mean = subthreshold_adaptation_strength * dt / tau_adap_mean
    if (subthreshold_adaptation_strength_mean is not None
            and tau_adap_mean is None):
        fa2_mean = subthreshold_adaptation_strength_mean * dt / tau_adap
    if (subthreshold_adaptation_strength_mean is not None
            and tau_adap_mean is not None):
        fa2_mean = subthreshold_adaptation_strength_mean * dt / tau_adap_mean

    # Convert to tensors
    leak = torch.as_tensor(leak, device=dev)
    g_l = torch.as_tensor(g_l, device=dev)
    g_l_mean = torch.as_tensor(g_l_mean, device=dev) if g_l_mean is not None \
        else None
    threshold = torch.as_tensor(threshold, device=dev)
    threshold_mean = torch.as_tensor(threshold_mean, device=dev) \
        if threshold_mean is not None else None
    reset = torch.as_tensor(reset, device=dev)
    tau_syn = torch.as_tensor(tau_syn, device=dev)
    c_mem = torch.as_tensor(c_mem, device=dev)
    exp_slope = torch.as_tensor(exp_slope, device=dev)
    exp_threshold = torch.as_tensor(exp_threshold, device=dev)
    subthreshold_adaptation_strength = torch.as_tensor(
        subthreshold_adaptation_strength, device=dev)
    spike_triggered_adaptation_increment = torch.as_tensor(
        spike_triggered_adaptation_increment, device=dev)
    spike_triggered_adaptation_increment_mean = torch.as_tensor(
        spike_triggered_adaptation_increment_mean, device=dev) if \
        spike_triggered_adaptation_increment_mean is not None else None
    tau_adap = torch.as_tensor(tau_adap, device=dev)
    fi1 = torch.as_tensor(fi1, device=dev)
    fv1 = torch.as_tensor(fv1, device=dev)
    fv2 = torch.as_tensor(fv2, device=dev)
    fv3 = torch.as_tensor(fv3, device=dev)
    fa1 = torch.as_tensor(fa1, device=dev)
    fa2 = torch.as_tensor(fa2, device=dev)
    fi1_mean = torch.as_tensor(fi1_mean, device=dev) if fi1_mean is not None \
        else None
    fv1_mean = torch.as_tensor(fv1_mean, device=dev) if fv1_mean is not None \
        else None
    fv2_mean = torch.as_tensor(fv2_mean, device=dev) if fv2_mean is not None \
        else None
    fv3_mean = torch.as_tensor(fv3_mean, device=dev) if fv3_mean is not None \
        else None
    fa1_mean = torch.as_tensor(fa1_mean, device=dev) if fa1_mean is not None \
        else None
    fa2_mean = torch.as_tensor(fa2_mean, device=dev) if fa2_mean is not None \
        else None

    current = torch.empty_like(input, device=dev)
    spikes = torch.empty_like(input, device=dev)
    membrane = torch.empty_like(input, device=dev)
    adaptation_storage = torch.empty_like(input, device=dev)

    # Set up mock attributes
    if trace_noise_current is None:
        trace_noise_current = RandomNoise(None, dev)
    if trace_noise_voltage is None:
        trace_noise_voltage = RandomNoise(None, dev)
    if trace_noise_adaptation is None:
        trace_noise_adaptation = RandomNoise(None, dev)
    trace_noise_current.to(dev)
    trace_noise_voltage.to(dev)
    trace_noise_adaptation.to(dev)

    lower_limit = -torch.inf
    upper_limit = torch.inf
    if dynamic_range_current is None:
        dynamic_range_current = Bounds(lower_limit, upper_limit)
    if dynamic_range_voltage is None:
        dynamic_range_voltage = Bounds(lower_limit, upper_limit)
    if dynamic_range_adaptation is None:
        dynamic_range_adaptation = Bounds(lower_limit, upper_limit)
    dynamic_range_current.to(dev)
    dynamic_range_voltage.to(dev)
    dynamic_range_adaptation.to(dev)

    # Initialize dictionary to match locals with code variables
    variables = {'i': i, 'z': z, 'adaptation': adaptation,
                 'dt': dt, 'fi1': fi1,
                 'fi1_mean': fi1_mean, 'fv1': fv1, 'fv1_mean': fv1_mean,
                 'exp': torch.exp, 'Unterjubel': Unterjubel,
                 'random_noise_current': trace_noise_current,
                 'random_noise_voltage': trace_noise_voltage,
                 'random_noise_adaptation': trace_noise_adaptation,
                 'RandomNoiseAdd': RandomNoiseAdd,
                 'dynamic_range_current': dynamic_range_current,
                 'dynamic_range_voltage': dynamic_range_voltage,
                 'dynamic_range_adaptation': dynamic_range_adaptation,
                 'saturate': saturate,
                 'FPNM': FixedPatternNoiseMultiply}

    if leaky:
        v[:] = leak
        variables['leak'] = leak
        if g_l is None:
            g_l = torch.ones(ps)
        variables['g_l'] = g_l
        variables['g_l_mean'] = g_l_mean
    else:
        v[:, :] = 0.
    if fire:
        variables['spike_surrogate'] = spike_surrogate
        variables['threshold'] = threshold
        variables['threshold_mean'] = threshold_mean
        variables['reset'] = reset
    if exponential:
        variables['fv2'] = fv2
        variables['fv2_mean'] = fv2_mean
        variables['fv3'] = fv3
        variables['fv3_mean'] = fv3_mean
        variables['exp_threshold'] = exp_threshold
        variables['exp_threshold_mean'] = exp_threshold_mean
        variables['exp_slope'] = exp_slope
        variables['exp_slope_mean'] = exp_slope_mean
        variables['FPNE'] = FixedPatternNoiseExponential
    adaptation_flag = spike_triggered_adaptation or subthreshold_adaptation
    if adaptation_flag:
        variables['fa1'] = fa1
        variables['fa1_mean'] = fa1_mean
    if subthreshold_adaptation:
        v[:] = leak
        variables['leak'] = leak
        variables['fa2'] = fa2
        variables['fa2_mean'] = fa2_mean
    if spike_triggered_adaptation:
        variables['b'] = spike_triggered_adaptation_increment
        variables['b_mean'] = spike_triggered_adaptation_increment_mean
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

    # Add CADC readout noise
    if cadc_readout_noise_current is not None:
        current = RandomNoiseAdd.apply(
            current, cadc_readout_noise_current.sample(current.shape))
    if cadc_readout_noise_voltage is not None:
        if membrane_cadc_hw is None:
            membrane = RandomNoiseAdd.apply(
                membrane, cadc_readout_noise_voltage.sample(membrane.shape))
    if cadc_readout_noise_adaptation is not None:
        if adaptation_cadc_hw is None:
            adaptation_storage = RandomNoiseAdd.apply(
                adaptation_storage,
                cadc_readout_noise_adaptation.sample(adaptation_storage.shape))

    # Clamp results at CADC readout bounds
    if cadc_readout_bounds_current is not None:
        membrane = saturate(membrane, cadc_readout_bounds_current)
    if cadc_readout_bounds_voltage is not None:
        membrane = saturate(membrane, cadc_readout_bounds_voltage)
    if cadc_readout_bounds_adaptation is not None:
        adaptation_storage = saturate(
            adaptation_storage, cadc_readout_bounds_adaptation)

    return (membrane, membrane_madc_hw, current, current_exc_cadc_hw,
            current_exc_madc_hw, current_inh_cadc_hw, current_inh_madc_hw,
            adaptation_storage, adaptation_madc_hw, spikes)
