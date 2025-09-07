"""
Test CuBa AELIF integration function
"""
import unittest
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from hxtorch.spiking.functional import cuba_aelif_integration, CuBaStepCode
from hxtorch.spiking import Handle
from hxtorch.spiking.observables import AnalogObservable

class TestAELIFIntegration(unittest.TestCase):
    """
    Test current based AELIF integration function.
    Test several different model configurations.
    Check output types depending on the configuration and plot simulation
    results for manual sanity check.
    """

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok = True)

    def test_cuba_aelif_integration(self):
        """
        Test CuBa AELIF integration function.
        """

        population_size = 15
        batch_size = 10
        time_steps = 100
        dt = 1e-6

        # Parameters
        leak = torch.Tensor([0]).expand(population_size)
        reset = torch.Tensor([-0.1]).expand(population_size)
        threshold = torch.Tensor([0.7]).expand(population_size)
        tau_syn = torch.Tensor([10e-6]).expand(population_size)
        c_mem = torch.Tensor([10e-6]).expand(population_size)
        g_l = torch.Tensor([1]).expand(population_size)
        refractory_time = torch.Tensor([3e-6]).expand(population_size)
        method = "superspike"
        alpha = 50
        exp_slope = torch.Tensor([200e-3]).expand(population_size)
        exp_threshold = torch.Tensor([0.3]).expand(population_size)
        subthreshold_adaptation_strength = torch.Tensor([10]).expand(
            population_size)
        spike_triggered_adaptation_increment = torch.Tensor([0.2]).expand(
            population_size)
        tau_adap = torch.Tensor([100e-6]).expand(population_size)

        # Inputs
        inputs = torch.zeros(time_steps, batch_size, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 2:4] = 1
        inputs[40, :, 1] = 1
        inputs[53, :, 4] = 1

        weights = 1 * \
            torch.nn.parameter.Parameter(torch.randn(population_size, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weights)

        # Test full AELIF model (all options enabled)
        integration_step_code = CuBaStepCode(
            leaky=True, fire=True, refractory=True, exponential=True,
            subthreshold_adaptation=True, spike_triggered_adaptation=True,
            hw_voltage_trace_available=False,
            hw_adaptation_trace_available=False, hw_spikes_available=False).\
            generate()
        self.assertTrue(type(integration_step_code) == str)
        aelif_data = cuba_aelif_integration(
            graded_spikes, leak=leak, reset=reset, threshold=threshold,
            tau_syn=tau_syn, c_mem=c_mem, g_l=g_l,
            refractory_time=refractory_time, method=method, alpha=alpha,
            exp_slope=exp_slope, exp_threshold=exp_threshold,
            subthreshold_adaptation_strength=subthreshold_adaptation_strength,
            spike_triggered_adaptation_increment=(
                spike_triggered_adaptation_increment),
            tau_adap=tau_adap, dt=dt, leaky=True, fire=True, refractory=True,
            exponential=True, subthreshold_adaptation=True,
            spike_triggered_adaptation=True,
            integration_step_code=integration_step_code)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([time_steps, batch_size, population_size]),
                torch.tensor(aelif_data.membrane_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([time_steps, batch_size, population_size]),
                torch.tensor(aelif_data.current.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([time_steps, batch_size, population_size]),
                torch.tensor(aelif_data.adaptation_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([time_steps, batch_size, population_size]),
                torch.tensor(aelif_data.spikes.shape)))
        self.assertIsNone(aelif_data.membrane_madc)
        self.assertIsNone(aelif_data.adaptation_madc)

        # No backpropagation error
        loss = aelif_data.spikes.sum()
        loss.backward()

        # Plot voltage, current and adaptation of first neuron
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., dt * time_steps, dt),
            aelif_data.membrane_cadc[:, 0, 0].detach().numpy(),
            label='membrane')
        ax.plot(
            np.arange(0., dt * time_steps, dt),
            aelif_data.current[:, 0, 0].detach().numpy(), label='current')
        ax.plot(
            np.arange(0., dt * time_steps, dt),
            aelif_data.adaptation_cadc[:, 0, 0].detach().numpy(),
            label='adaptation')
        ax.axhline(exp_threshold[0], label='exp_threshold', linestyle='--',
                   color='red')
        ax.legend()

        plt.savefig(self.plot_path.joinpath("./cuba_aelif_dynamics_mock.png"))

    def test_cuba_aelif_integration_hw_data(self):
        """
        Test CuBa AELIF integration function with artificially generated
        hardware data.
        """

        population_size = 15
        batch_size = 10
        time_steps = 100
        dt = 1e-6

        # Parameters
        leak = torch.Tensor([0]).expand(population_size)
        reset = torch.Tensor([-0.1]).expand(population_size)
        threshold = torch.Tensor([0.7]).expand(population_size)
        tau_syn = torch.Tensor([10e-6]).expand(population_size)
        c_mem = torch.Tensor([10e-6]).expand(population_size)
        g_l = torch.Tensor([1]).expand(population_size)
        refractory_time = torch.Tensor([3e-6]).expand(population_size)
        method = "superspike"
        alpha = 50
        exp_slope = torch.Tensor([50e-3]).expand(population_size)
        exp_threshold = torch.Tensor([0.3]).expand(population_size)
        subthreshold_adaptation_strength = torch.Tensor([1]).expand(
            population_size)
        spike_triggered_adaptation_increment = torch.Tensor([0.5]).expand(
            population_size)
        tau_adap = torch.Tensor([100e-6]).expand(population_size)

        # Inputs
        inputs = torch.zeros(time_steps, batch_size, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 2:4] = 1
        inputs[40, :, 1] = 1
        inputs[53, :, 4] = 1

        weights = 0.1 * \
            torch.nn.parameter.Parameter(torch.randn(population_size, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weights)

        # Generate observable data which is to be injected as hardware data
        integration_step_code = CuBaStepCode(
            leaky=True, fire=True, refractory=True, exponential=True,
            subthreshold_adaptation=True, spike_triggered_adaptation=True,
            hw_voltage_trace_available=False, hw_spikes_available=False).\
            generate()
        hw_data = cuba_aelif_integration(
            graded_spikes, leak=leak, reset=reset, threshold=threshold,
            tau_syn=tau_syn, c_mem=c_mem, g_l=g_l,
            refractory_time=refractory_time, method=method, alpha=alpha,
            exp_slope=exp_slope, exp_threshold=exp_threshold,
            subthreshold_adaptation_strength=subthreshold_adaptation_strength,
            spike_triggered_adaptation_increment=(
                spike_triggered_adaptation_increment),
            tau_adap=tau_adap, dt=dt, leaky=True, fire=True, refractory=True,
            exponential=True, subthreshold_adaptation=True,
            spike_triggered_adaptation=True,
            integration_step_code=integration_step_code)
        self.assertIsNone(hw_data.membrane_madc)
        self.assertIsNone(hw_data.adaptation_madc)

        # Add jitter
        hw_voltage = hw_data.membrane_cadc + \
            torch.rand(hw_data.membrane_cadc.shape) * 0.05
        hw_adaptation = hw_data.adaptation_cadc + \
            torch.rand(hw_data.adaptation_cadc.shape) * 0.05
        hw_spikes = hw_data.spikes
        hw_spikes[
            torch.randint(time_steps, (1,)), torch.randint(batch_size, (1,)),
            torch.randint(population_size, (1,))] = 1
        injected_hw_data = Handle(
            voltage=AnalogObservable(cadc=hw_voltage, madc=hw_adaptation),
            adaptation=AnalogObservable(cadc=hw_adaptation, madc=None),
            spikes=hw_spikes)

        # Inject hw_data into new integration process
        integration_step_code_hw = CuBaStepCode(
            leaky=True, fire=True, refractory=True, exponential=True,
            subthreshold_adaptation=True, spike_triggered_adaptation=True,
            hw_voltage_trace_available=True,
            hw_adaptation_trace_available=True, hw_spikes_available=True).\
            generate()
        aelif_data = cuba_aelif_integration(
            graded_spikes, leak=leak, reset=reset, threshold=threshold,
            tau_syn=tau_syn, c_mem=c_mem, g_l=g_l,
            refractory_time=refractory_time, method=method, alpha=alpha,
            exp_slope=exp_slope, exp_threshold=exp_threshold,
            subthreshold_adaptation_strength=subthreshold_adaptation_strength,
            spike_triggered_adaptation_increment=(
                spike_triggered_adaptation_increment),
            tau_adap=tau_adap, hw_data=injected_hw_data,
            dt=dt, leaky=True, fire=True, refractory=True, exponential=True,
            subthreshold_adaptation=True, spike_triggered_adaptation=True,
            integration_step_code=integration_step_code_hw)

        # Check if injected hardware data is still the same
        self.assertTrue(torch.equal(hw_voltage, aelif_data.membrane_cadc))
        self.assertTrue(torch.equal(hw_adaptation, aelif_data.membrane_madc))
        self.assertTrue(
            torch.equal(hw_adaptation, aelif_data.adaptation_cadc))
        self.assertIsNone(aelif_data.adaptation_madc)
        self.assertTrue(torch.equal(hw_spikes, aelif_data.spikes))

        # Shapes of remaining observables
        self.assertTrue(
            torch.equal(
                torch.tensor(hw_voltage.shape),
                torch.tensor(aelif_data.current.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor(hw_voltage.shape),
                torch.tensor(aelif_data.adaptation_cadc.shape)))

        # No backpropagation error
        loss = aelif_data.spikes.sum()
        loss.backward()

        # Plot voltage, current and adaptation of first neuron
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., dt * time_steps, dt),
            aelif_data.membrane_cadc[:, 0, 0].detach().numpy(),
            label='membrane')
        ax.plot(
            np.arange(0., dt * time_steps, dt),
            aelif_data.current[:, 0, 0].detach().numpy(), label='current')
        ax.plot(
            np.arange(0., dt * time_steps, dt),
            aelif_data.adaptation_cadc[:, 0, 0].detach().numpy(),
            label='adaptation')
        ax.legend()

        plt.savefig(self.plot_path.joinpath("./cuba_aelif_dynamics_hw.png"))


    def test_error_message(self):
        """
        Test custom error handling done in the CuBa AELIF integration function.
        """

        population_size = 15
        batch_size = 10
        time_steps = 100
        dt = 1e-6

        # Parameters
        leak = torch.Tensor([0]).expand(population_size)
        reset = torch.Tensor([-0.1]).expand(population_size)
        threshold = torch.Tensor([0.7]).expand(population_size)
        tau_syn = torch.Tensor([10e-6]).expand(population_size)
        c_mem = torch.Tensor([10e-6]).expand(population_size)
        g_l = torch.Tensor([1]).expand(population_size)
        refractory_time = torch.Tensor([3e-6]).expand(population_size)
        method = "superspike"
        alpha = 50
        exp_slope = torch.Tensor([50e-3]).expand(population_size)
        exp_threshold = torch.Tensor([0.3]).expand(population_size)
        subthreshold_adaptation_strength = torch.Tensor([1]).expand(
            population_size)
        spike_triggered_adaptation_increment = torch.Tensor([0.5]).expand(
            population_size)
        tau_adap = torch.Tensor([100e-6]).expand(population_size)

        # Inputs
        inputs = torch.zeros(time_steps, batch_size, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 2:4] = 1
        inputs[40, :, 1] = 1
        inputs[53, :, 4] = 1

        weights = 0.1 * \
            torch.nn.parameter.Parameter(torch.randn(population_size, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weights)

        # Generate observable data which is to be injected as hardware data
        integration_step_code = CuBaStepCode(
            leaky=True, fire=True, refractory=True, exponential=True,
            subthreshold_adaptation=True, spike_triggered_adaptation=True,
            hw_voltage_trace_available=False, hw_spikes_available=False).\
            generate()

        # Add error source to integration_step_code
        constructed_error_msg = "Test succeeded, if this is shown!"
        error_line = f"raise RuntimeError(\"{constructed_error_msg}\")"
        integration_step_code = integration_step_code + "\n" + error_line
        lineno = len(integration_step_code.splitlines())
        expected_error_msg = (
            "An error occured while executing the code of the integration"
            + f" step.\nIn line {lineno} (\"{error_line}\") in"
            + " integration_step_code, following error occured:\n"
            + constructed_error_msg)

        # Examine error
        with self.assertRaises(RuntimeError) as context:
            hw_data = cuba_aelif_integration(
                graded_spikes, leak=leak, reset=reset, threshold=threshold,
                tau_syn=tau_syn, c_mem=c_mem, g_l=g_l,
                refractory_time=refractory_time, method=method, alpha=alpha,
                exp_slope=exp_slope, exp_threshold=exp_threshold,
                subthreshold_adaptation_strength=(
                    subthreshold_adaptation_strength),
                spike_triggered_adaptation_increment=(
                    spike_triggered_adaptation_increment),
                tau_adap=tau_adap, dt=dt, leaky=True, fire=True,
                refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=integration_step_code)
        exception = context.exception

        # Compare error messages
        self.assertEqual(str(exception), expected_error_msg)


if __name__ == "__main__":
    unittest.main()
