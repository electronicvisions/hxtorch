from textwrap import dedent


class CuBaStepCode():
    # Initialize code snippets
    base_code = dedent(
        """
        # Membrane increment
        dv = dt / c * (i{leak_term}{exponential_term}{adaptation_v_dgl})

        # Current
        i = i * (1 - dt / tau_syn) + input[ts]

        # Apply integration step
        {unterjubel_voltage}

        # Spikes
        {spike}
        {unterjubel_spike}

        # Adaptation
        {adaptation_adaptation_dgl}{subthreshold_adaptation}\
        {spike_triggered_adaptation}
        {unterjubel_adaptation}

        # Reset
        {non_unterjubel_reset}

        # Refractory update
        {refractory_update}
        """)

    leak_term_code = " + g_l * (leak - v)"

    exponential_code = \
        " + g_l * exp_slope * exp((v - exp_threshold) / exp_slope)"

    adaptation_code_v_dgl = " - adaptation"

    unterjubel_v_code = "v = Unterjubel.apply(dv + v, membrane_hw[ts])"

    non_unterjubel_v_code = "v = v + dv"

    spike_code = \
        "spike = spiking_threshold(v - threshold, method, alpha)"

    unterjubel_z_code = "z = Unterjubel.apply(spike, spikes_hw[ts])"

    non_unterjubel_z_code = "z = spike"

    adaptation_code_adaptation_dgl = \
        "adaptation = adaptation * (1 - dt / tau_adaptation)"

    subthreshold_adaptation_code = " + dt / tau_adaptation * a * (v - leak)"

    spike_triggered_adaptation_code = " + b * z"

    unterjubel_adaptation_code = \
        "adaptation = Unterjubel.apply(adaptation, adaptation_hw[ts])"

    non_unterjubel_reset_code = "v = (1 - z.detach()) * v + z.detach() * reset"

    refractory_update = \
        """z, v, ref_state = refractory_update(
    z, v, ref_state, spikes_hw[ts], membrane_hw[ts],
    refractory_time=refractory_time, reset=reset, dt=dt)"""

    # pylint: disable=too-many-arguments
    def __init__(self, leaky: bool = True,
                 fire: bool = True,
                 refractory: bool = False,
                 exponential: bool = False,
                 subthreshold_adaptation: bool = False,
                 spike_triggered_adaptation: bool = False,
                 hw_voltage_trace_available: bool = False,
                 hw_adaptation_trace_available: bool = False,
                 hw_spikes_available: bool = False) -> None:
        """
        Initialize a code factory that can generate the necessary
        code for executing one integration step in simulation of the forward
        pass.
        Based on the passed arguments, subparts of the AdEx differential
        equations are picked to be simulated. However, the least complex
        version still includes all terms necessary to simulate a leaky
        integrator (LI).
        Per default, the leaky-integrate-and-fire model (LIF) is simulated.

        :param leaky: Flag that enables / disables the leak term when set
            to true / false
        :param fire: Flag that enables / disables firing behaviour when set
            to true / false.
        :param refractory: Flag that is used to omit the execution of the
            refractory update in case the refractory time is set to zero.
        :param exponential: Flag that enables / disables the exponential term
            in the differential equation for the membrane potential when set
            to true / false.
        :param subthreshold_adaptation: Flag that enables / disables the
            subthreshold adaptation term in the differential equation of the
            adaptation when set to true / false.
        :param spike_triggered_adaptation: Flag that enables / disables
            spike-triggered adaptation when set to true / false.

        If neither `subthreshold_adaptation` nor `spike_triggered_adaptation`
        are enabled, the adaptation won't be simulated at all.
        """
        # Initialize flags
        self.leaky = leaky
        self.fire = fire
        self.refractory = refractory
        self.exponential = exponential
        self.subthreshold_adaptation = subthreshold_adaptation
        self.spike_triggered_adaptation = spike_triggered_adaptation
        self.adaptation = self.subthreshold_adaptation \
            or self.spike_triggered_adaptation
        self.hw_voltage_trace_available = hw_voltage_trace_available
        self.hw_adaptation_trace_available = hw_adaptation_trace_available
        self.hw_spikes_available = hw_spikes_available

    def generate(self) -> str:
        leak_term = self.leak_term_code if self.leaky else ""
        exponential_term = self.exponential_code if self.exponential else ""
        adaptation_v_dgl = self.adaptation_code_v_dgl if self.adaptation \
            else ""
        unterjubel_voltage = self.unterjubel_v_code if \
            self.hw_voltage_trace_available else self.non_unterjubel_v_code
        spike = self.spike_code if self.fire else ""
        unterjubel_spike = ""
        if self.fire:
            unterjubel_spike = self.unterjubel_z_code if \
                self.hw_spikes_available else self.non_unterjubel_z_code
        adaptation_adaptation_dgl = self.adaptation_code_adaptation_dgl if \
            self.adaptation else ""
        subthreshold_adaptation = self.subthreshold_adaptation_code if \
            self.subthreshold_adaptation else ""
        spike_triggered_adaptation = self.spike_triggered_adaptation_code if \
            self.spike_triggered_adaptation else ""
        unterjubel_adaptation = self.unterjubel_adaptation_code if \
            self.hw_adaptation_trace_available else ""
        non_unterjubel_reset = self.non_unterjubel_reset_code if \
            self.fire and not self.hw_voltage_trace_available else ""
        refractory_update = self.refractory_update if self.refractory else ""

        generated_code = self.base_code.format(
            leak_term=leak_term,
            exponential_term=exponential_term,
            adaptation_v_dgl=adaptation_v_dgl,
            unterjubel_voltage=unterjubel_voltage,
            spike=spike,
            unterjubel_spike=unterjubel_spike,
            adaptation_adaptation_dgl=adaptation_adaptation_dgl,
            subthreshold_adaptation=subthreshold_adaptation,
            spike_triggered_adaptation=spike_triggered_adaptation,
            unterjubel_adaptation=unterjubel_adaptation,
            non_unterjubel_reset=non_unterjubel_reset,
            refractory_update=refractory_update)

        return generated_code
