"""
Test CuBa AELIF integration function
"""
import unittest
from pathlib import Path
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt

from hxtorch.spiking.functional import cuba_aelif_integration, CuBaStepCode
from hxtorch.spiking import Handle
from hxtorch.spiking.observables import AnalogObservable
from hxtorch.spiking.functional.surrogates import superspike


class TestAELIFIntegration(unittest.TestCase):
    """
    Test current based AELIF integration function.
    Test several different model configurations.
    Check output types depending on the configuration and plot simulation
    results for manual sanity check.
    """

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

        self.population_size = 15
        self.batch_size = 10
        self.time_steps = 100
        self.dt = 1e-6

        # Parameters
        self.leak = torch.Tensor([0]).expand(self.population_size)
        self.reset = torch.Tensor([-0.1]).expand(self.population_size)
        self.threshold = torch.Tensor([0.7]).expand(self.population_size)
        self.tau_syn = torch.Tensor([10e-6]).expand(self.population_size)
        self.c_mem = torch.Tensor([10e-6]).expand(self.population_size)
        self.g_l = torch.Tensor([1]).expand(self.population_size)
        self.refractory_time = torch.Tensor([3e-6]).expand(
            self.population_size)
        self.spike_surrogate = partial(superspike, alpha=50)
        self.exp_slope = torch.Tensor([200e-3]).expand(self.population_size)
        self.exp_threshold = torch.Tensor([0.3]).expand(self.population_size)
        self.subthreshold_adaptation_strength = torch.Tensor([10]).expand(
            self.population_size)
        self.spike_triggered_adaptation_increment = torch.Tensor([0.2]).expand(
            self.population_size)
        self.tau_adap = torch.Tensor([100e-6]).expand(self.population_size)

        # Inputs
        inputs = torch.zeros(self.time_steps, self.batch_size, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 2:4] = 1
        inputs[40, :, 1] = 1
        inputs[53, :, 4] = 1

        torch.manual_seed(42)
        weights = 1 * \
            torch.nn.parameter.Parameter(torch.randn(self.population_size, 5))
        self.graded_spikes = torch.nn.functional.linear(inputs, weights)

    def test_cuba_aelif_integration(self):
        """
        Test CuBa AELIF integration function.
        """

        # Test full AELIF model (all options enabled)
        integration_step_code = CuBaStepCode(
            leaky=True, fire=True, refractory=True, exponential=True,
            subthreshold_adaptation=True, spike_triggered_adaptation=True,
            hw_voltage_trace_available=False,
            hw_adaptation_trace_available=False, hw_spikes_available=False).\
            generate()
        self.assertTrue(type(integration_step_code) == str)
        (membrane_cadc, membrane_madc, current, adaptation_cadc,
            adaptation_madc, spikes) = cuba_aelif_integration(
                self.graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, tau_syn=self.tau_syn,
                c_mem=self.c_mem, g_l=self.g_l,
                refractory_time=self.refractory_time,
                spike_surrogate=self.spike_surrogate,
                exp_slope=self.exp_slope, exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    self.subthreshold_adaptation_strength),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, dt=self.dt, leaky=True, fire=True,
                refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=integration_step_code)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([self.time_steps, self.batch_size,
                              self.population_size]),
                torch.tensor(membrane_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([self.time_steps, self.batch_size,
                              self.population_size]),
                torch.tensor(current.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([self.time_steps, self.batch_size,
                              self.population_size]),
                torch.tensor(adaptation_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([self.time_steps, self.batch_size,
                              self.population_size]),
                torch.tensor(spikes.shape)))
        self.assertIsNone(membrane_madc)
        self.assertIsNone(adaptation_madc)

        # No backpropagation error
        loss = spikes.sum()
        loss.backward()

        # Plot voltage, current and adaptation of first neuron
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., self.dt * self.time_steps, self.dt),
            membrane_cadc[:, 0, 0].detach().numpy(),
            label='membrane')
        ax.plot(
            np.arange(0., self.dt * self.time_steps, self.dt),
            current[:, 0, 0].detach().numpy(), label='current')
        ax.plot(
            np.arange(0., self.dt * self.time_steps, self.dt),
            adaptation_cadc[:, 0, 0].detach().numpy(),
            label='adaptation')
        ax.axhline(self.exp_threshold[0], label='exp_threshold', linestyle='--',
                   color='red')
        ax.legend()

        plt.savefig(self.plot_path.joinpath("./cuba_aelif_dynamics_mock.png"))

    def test_cuba_aelif_integration_hw_data(self):
        """
        Test CuBa AELIF integration function with artificially generated
        hardware data.
        """

        # Generate observable data which is to be injected as hardware data
        integration_step_code = CuBaStepCode(
            leaky=True, fire=True, refractory=True, exponential=True,
            subthreshold_adaptation=True, spike_triggered_adaptation=True,
            hw_voltage_trace_available=False, hw_spikes_available=False).\
            generate()
        (membrane_cadc_hw, membrane_madc_hw, current_hw, adaptation_cadc_hw,
            adaptation_madc_hw, spikes_hw) = cuba_aelif_integration(
                self.graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, tau_syn=self.tau_syn,
                c_mem=self.c_mem, g_l=self.g_l,
                refractory_time=self.refractory_time,
                spike_surrogate=self.spike_surrogate,
                exp_slope=self.exp_slope, exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    self.subthreshold_adaptation_strength),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, dt=self.dt, leaky=True, fire=True,
                refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=integration_step_code)
        self.assertIsNone(membrane_madc_hw)
        self.assertIsNone(adaptation_madc_hw)

        # Add jitter
        hw_voltage = membrane_cadc_hw + \
            torch.rand(membrane_cadc_hw.shape) * 0.05
        hw_adaptation = adaptation_cadc_hw + \
            torch.rand(adaptation_cadc_hw.shape) * 0.05
        hw_spikes = spikes_hw
        hw_spikes[
            torch.randint(self.time_steps, (1,)), torch.randint(self.batch_size, (1,)),
            torch.randint(self.population_size, (1,))] = 1
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
        (membrane_cadc, membrane_madc, current, adaptation_cadc,
            adaptation_madc, spikes) = cuba_aelif_integration(
                self.graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, tau_syn=self.tau_syn,
                c_mem=self.c_mem, g_l=self.g_l,
                refractory_time=self.refractory_time,
                spike_surrogate=self.spike_surrogate,
                exp_slope=self.exp_slope, exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    self.subthreshold_adaptation_strength),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, hw_data=injected_hw_data,
                dt=self.dt, leaky=True, fire=True, refractory=True,
                exponential=True, subthreshold_adaptation=True,
                spike_triggered_adaptation=True,
                integration_step_code=integration_step_code_hw)

        # Check if injected hardware data is still the same
        self.assertTrue(torch.equal(hw_voltage, membrane_cadc))
        self.assertTrue(torch.equal(hw_adaptation, membrane_madc))
        self.assertTrue(
            torch.equal(hw_adaptation, adaptation_cadc))
        self.assertIsNone(adaptation_madc)
        self.assertTrue(torch.equal(hw_spikes, spikes))

        # Shapes of remaining observables
        self.assertTrue(
            torch.equal(
                torch.tensor(hw_voltage.shape),
                torch.tensor(current.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor(hw_voltage.shape),
                torch.tensor(adaptation_cadc.shape)))

        # No backpropagation error
        loss = spikes.sum()
        loss.backward()

        # Plot voltage, current and adaptation of first neuron
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., self.dt * self.time_steps, self.dt),
            membrane_cadc[:, 0, 0].detach().numpy(),
            label='membrane')
        ax.plot(
            np.arange(0., self.dt * self.time_steps, self.dt),
            current[:, 0, 0].detach().numpy(), label='current')
        ax.plot(
            np.arange(0., self.dt * self.time_steps, self.dt),
            adaptation_cadc[:, 0, 0].detach().numpy(),
            label='adaptation')
        ax.axhline(self.exp_threshold[0], label='exp_threshold', linestyle='--',
                   color='red')
        ax.legend()

        plt.savefig(self.plot_path.joinpath("./cuba_aelif_dynamics_hw.png"))

    def test_error_message(self):
        """
        Test custom error handling done in the CuBa AELIF integration function.
        """

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
                self.graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, tau_syn=self.tau_syn,
                c_mem=self.c_mem, g_l=self.g_l,
                refractory_time=self.refractory_time,
                spike_surrogate=self.spike_surrogate,
                exp_slope=self.exp_slope, exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    self.subthreshold_adaptation_strength),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, dt=self.dt, leaky=True, fire=True,
                refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=integration_step_code)
        exception = context.exception

        # Compare error messages
        self.assertEqual(str(exception), expected_error_msg)


if __name__ == "__main__":
    unittest.main()
