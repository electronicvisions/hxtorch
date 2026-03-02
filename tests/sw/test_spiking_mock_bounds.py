"""
Test mocking of saturation effects triggered by a finite readout range or
dynamic ranges.
"""
import unittest
from pathlib import Path
from functools import partial

import numpy as np
import torch
import matplotlib.pyplot as plt

from hxtorch.spiking.functional.mock import Bounds
from hxtorch.spiking.functional.surrogates import (
    superspike, exponential_rolloff)
import hxtorch.spiking as hxsnn


# pylint: disable=too-many-instance-attributes
class TestBounds(unittest.TestCase):
    """
    Test mocking of saturation at bounds
    """
    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        # Plot path
        self.plot_path.mkdir(exist_ok=True)
        # Set population size
        self.size = 10
        # Inputs
        self.inputs = torch.zeros(100, 5, 10)
        self.inputs[10, :, 0] = 1
        self.inputs[30, :, 1] = 1
        self.inputs[40, :, 2] = 1
        self.inputs[53, :, 3] = 1
        # Neuron Parameters
        self.tau_syn = 10e-6
        self.tau_mem = 10e-6
        self.leak = 0.
        self.reset = 0.
        self.threshold = 1.
        self.g_l = 1.
        self.refractory_time = 1e-6
        self.spike_surrogate = partial(superspike, alpha=50)
        self.exp_slope = 0.2
        self.exp_threshold = 0.3
        self.spike_triggered_adaptation_increment = 0.2
        self.subthreshold_adaptation_strength = 2.
        self.tau_adap = 100e-6
        # Integration step code
        self.integration_step_code = hxsnn.functional.CuBaStepCode(
            leaky=True, fire=True, refractory=True, exponential=True,
            subthreshold_adaptation=True, spike_triggered_adaptation=True,
            hw_voltage_trace_available=False,
            hw_adaptation_trace_available=False, hw_spikes_available=False,
            finite_dynamic_ranges=True).generate()
        # Set random seed for reproducability
        torch.manual_seed(42)

    def test_bounds(self):
        """
        Test, if functionalities of Bounds work, as intended.
        """
        # Test device assignment of instance attributes
        cpu = torch.device("cpu")
        bounds = Bounds(lower=-0.5, upper=1.5, device=cpu)
        self.assertEqual(bounds.lower.device, cpu)
        self.assertEqual(bounds.upper.device, cpu)
        if torch.cuda.is_available():
            gpu = torch.device("cuda")
            bounds.to(gpu)
            self.assertEqual(bounds.lower.device, gpu)
            self.assertEqual(bounds.upper.device, gpu)

        # Test default device
        bounds = Bounds(lower=-0.5, upper=1.5)
        self.assertEqual(bounds.lower.device, cpu)
        self.assertEqual(bounds.upper.device, cpu)

        # Test subscription
        explicit_bounds = Bounds(
            lower=torch.linspace(-0.7, -0.3, self.size),
            upper=torch.linspace(1.3, 1.7, self.size))
        crop_idx = int(self.size / 2)
        sliced_bounds = explicit_bounds[:crop_idx]
        self.assertTrue(torch.allclose(
            sliced_bounds.lower, explicit_bounds.lower[:crop_idx]))
        self.assertTrue(torch.allclose(
            sliced_bounds.upper, explicit_bounds.upper[:crop_idx]))

    def test_dynamic_range_bounds(self):
        """
        Simulate the trace of one neuron with and without bounds for the
        dynamic range of voltage and adaptation. Plot both of them in
        comparison and assert non-equality.
        """
        weights = torch.nn.parameter.Parameter(torch.ones(self.size, 10))
        graded_spikes = torch.nn.functional.linear(self.inputs, weights)

        # Generate traces without dynamic range bounds
        (membrane_cadc_clean_sim, _, current_cadc_clean_sim,
         adaptation_cadc_clean_sim, _, spikes_clean_sim) = \
            hxsnn.functional.cuba_aelif_integration(
                graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, c_mem=self.tau_mem,
                g_l=self.g_l, tau_syn=self.tau_syn,
                refractory_time=self.refractory_time,
                spike_surrogate=partial(superspike, alpha=50),
                exp_slope=self.exp_slope,
                exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    self.subthreshold_adaptation_strength),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, dt=2e-6, leaky=True, fire=True,
                refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=self.integration_step_code)

        # Generate traces with dynamic range bounds
        surrogate = partial(exponential_rolloff, rolloff_margin=0.05)
        dynamic_range_current = Bounds(
            -torch.inf, torch.inf, surrogate=surrogate)
        dynamic_range_voltage = Bounds(-0.5, 1.2, surrogate=surrogate)
        dynamic_range_adaptation = Bounds(-0.3, 0.6, surrogate=surrogate)
        (membrane_cadc_saturated_traces, _, current_cadc_saturated_traces,
         adaptation_cadc_saturated_traces, _, spikes_saturated_traces) = \
            hxsnn.functional.cuba_aelif_integration(
                graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, c_mem=self.tau_mem,
                g_l=self.g_l, tau_syn=self.tau_syn,
                refractory_time=self.refractory_time,
                spike_surrogate=partial(superspike, alpha=50),
                exp_slope=self.exp_slope, exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    self.subthreshold_adaptation_strength),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, dt=2e-6,
                dynamic_range_voltage=dynamic_range_voltage,
                dynamic_range_adaptation=dynamic_range_adaptation, leaky=True,
                fire=True, refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=self.integration_step_code)

        # Assert equality or non-equality respectively
        self.assertTrue(torch.allclose(
            current_cadc_clean_sim, current_cadc_saturated_traces))
        self.assertFalse(torch.allclose(
            membrane_cadc_clean_sim, membrane_cadc_saturated_traces))
        self.assertFalse(torch.allclose(
            adaptation_cadc_clean_sim, adaptation_cadc_saturated_traces))

        # Plot differing traces for manual visual control
        self.plot("dynamic_range_bounds.png", membrane_cadc_clean_sim,
                  adaptation_cadc_clean_sim, spikes_clean_sim,
                  membrane_cadc_saturated_traces,
                  adaptation_cadc_saturated_traces, spikes_saturated_traces,
                  "dynamic range bounds")

    def test_readout_range_bounds(self):
        """
        Simulate the trace of one neuron with and without bounds for the
        readout range of the CADC. Plot both of them in comparison and assert
        non-equality.
        """
        weights = torch.nn.parameter.Parameter(torch.ones(self.size, 10))
        graded_spikes = torch.nn.functional.linear(self.inputs, weights)

        # Generate traces without readout range bounds
        (membrane_cadc_clean_sim, _, current_cadc_clean_sim,
         adaptation_cadc_clean_sim, _, spikes_clean_sim) = \
            hxsnn.functional.cuba_aelif_integration(
                graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, c_mem=self.tau_mem,
                g_l=self.g_l, tau_syn=self.tau_syn,
                refractory_time=self.refractory_time,
                spike_surrogate=partial(superspike, alpha=50),
                exp_slope=self.exp_slope, exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    self.subthreshold_adaptation_strength),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, dt=2e-6, leaky=True, fire=True,
                refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=self.integration_step_code)

        # Generate traces with readout range bounds
        cadc_readout_bounds_current = Bounds(float('-inf'), float('inf'))
        cadc_readout_bounds_voltage = Bounds(-0.5, 0.6, hardware_aware=False)
        cadc_readout_bounds_adaptation = Bounds(
            -0.3, 0.5, hardware_aware=False)
        (membrane_cadc_saturated_traces, _, current_cadc_saturated_traces,
         adaptation_cadc_saturated_traces, _, spikes_saturated_traces) = \
            hxsnn.functional.cuba_aelif_integration(
                graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, c_mem=self.tau_mem,
                g_l=self.g_l, tau_syn=self.tau_syn,
                refractory_time=self.refractory_time,
                spike_surrogate=partial(superspike, alpha=50),
                exp_slope=self.exp_slope, exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    self.subthreshold_adaptation_strength),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, dt=2e-6,
                cadc_readout_bounds_voltage=cadc_readout_bounds_voltage,
                cadc_readout_bounds_adaptation=cadc_readout_bounds_adaptation,
                leaky=True, fire=True, refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=self.integration_step_code)

        # Assert equality or non-equality respectively
        self.assertTrue(torch.allclose(
            current_cadc_clean_sim, current_cadc_saturated_traces))
        self.assertFalse(torch.allclose(
            membrane_cadc_clean_sim, membrane_cadc_saturated_traces))
        self.assertFalse(torch.allclose(
            adaptation_cadc_clean_sim, adaptation_cadc_saturated_traces))

        # Plot differing traces for manual visual control
        self.plot("readout_range_bounds.png", membrane_cadc_clean_sim,
                  adaptation_cadc_clean_sim, spikes_clean_sim,
                  membrane_cadc_saturated_traces,
                  adaptation_cadc_saturated_traces, spikes_saturated_traces,
                  "readout range bounds")

    def test_hardware_aware_flag(self):
        """
        Simulate the trace of one neuron followed by hardware-aware and
        hardware-unaware backpropagation to verify that the `hardware_aware`
        flag of Bounds triggers a difference in the gradients.
        """
        # Generate traces and backpropagate hardware-awarely
        weights = torch.nn.parameter.Parameter(torch.ones(self.size, 10))
        graded_spikes = torch.nn.functional.linear(self.inputs, weights)
        cadc_readout_bounds_voltage = Bounds(-0.5, 0.6, hardware_aware=True)
        membrane_cadc_hardware_aware, _, _, _, _, _ = \
            hxsnn.functional.cuba_aelif_integration(
                graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, c_mem=self.tau_mem,
                g_l=self.g_l, tau_syn=self.tau_syn,
                refractory_time=self.refractory_time,
                spike_surrogate=self.spike_surrogate,
                exp_slope=self.exp_slope,
                exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    self.subthreshold_adaptation_strength),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, dt=2e-6,
                cadc_readout_bounds_voltage=cadc_readout_bounds_voltage,
                leaky=True, fire=True, refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=self.integration_step_code)

        # Compute backward pass
        hardware_aware_score = membrane_cadc_hardware_aware.sum()
        hardware_aware_score.backward()
        gradients_hardware_aware = weights.grad.detach().clone()

        # Generate traces and backpropagate hardware-unawarely
        weights = torch.nn.parameter.Parameter(torch.ones(self.size, 10))
        graded_spikes = torch.nn.functional.linear(self.inputs, weights)
        cadc_readout_bounds_voltage = Bounds(-0.5, 0.6, hardware_aware=False)
        membrane_cadc_hardware_unaware, _, _, _, _, _ = \
            hxsnn.functional.cuba_aelif_integration(
                graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, c_mem=self.tau_mem,
                g_l=self.g_l, tau_syn=self.tau_syn,
                refractory_time=self.refractory_time,
                spike_surrogate=partial(superspike, alpha=50),
                exp_slope=self.exp_slope,
                exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    self.subthreshold_adaptation_strength),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, dt=2e-6,
                cadc_readout_bounds_voltage=cadc_readout_bounds_voltage,
                leaky=True, fire=True, refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=self.integration_step_code)

        # Compute backward pass
        hardware_unaware_score = membrane_cadc_hardware_unaware.sum()
        hardware_unaware_score.backward()
        gradients_hardware_unaware = weights.grad.detach().clone()

        # Assert non-equality of gradients
        self.assertFalse(torch.allclose(
            gradients_hardware_aware, gradients_hardware_unaware))

    # pylint: disable=too-many-locals, too-many-arguments
    def plot(self, path: str, membrane_clean_sim: torch.Tensor,
             adaptation_clean_sim: torch.Tensor,
             spikes_clean_sim: torch.Tensor, membrane_mock: torch.Tensor,
             adaptation_mock: torch.Tensor, spikes_mock: torch.Tensor,
             hardware_specific_effect: str):
        """
        Plot voltage traces and spikes, as well as adaptation traces of both
        the simulation with and without certain hardware-specific effects.
        """
        membrane_sim = membrane_clean_sim[:, 0, 0].detach().numpy()
        membrane_mock = membrane_mock[:, 0, 0].detach().numpy()
        adaptation_sim = adaptation_clean_sim[:, 0, 0].detach().numpy()
        adaptation_mock = adaptation_mock[:, 0, 0].detach().numpy()
        spike_indices_sim = spikes_clean_sim.detach()[:, 0, 0].nonzero()[:, 0]
        spike_indices_mock = spikes_mock.detach()[:, 0, 0].nonzero()[:, 0]
        timesteps = np.arange(0, membrane_mock.shape[0], 1)
        fig = plt.figure(layout="tight", figsize=(8, 8))
        axs = fig.subplots(2, 1, sharex=False)
        axs[0].plot(timesteps, membrane_sim, alpha=0.5,
                    color='k', label="without " + hardware_specific_effect)
        axs[0].vlines(spike_indices_sim, ymin=-1e2, ymax=1e2,  color='red',
                      linewidth=1, alpha=0.5)
        axs[0].plot(timesteps, membrane_mock,
                    color='tab:blue', label="with " + hardware_specific_effect)
        axs[0].vlines(spike_indices_mock, ymin=-1e2, ymax=1e2, color='red',
                      linewidth=1., label="spikes")
        axs[0].set_xlabel("Timesteps")
        axs[0].legend()
        axs[0].set_xlim(0, timesteps.size)
        axs[0].set_ylim(1.3*np.min(membrane_mock), 1.3*np.max(membrane_mock))
        axs[0].set_title("Membrane trace")
        axs[1].plot(timesteps, adaptation_sim, alpha=0.5,
                    color='k', label="without " + hardware_specific_effect)
        axs[1].plot(timesteps, adaptation_mock, color='tab:orange',
                    label="with " + hardware_specific_effect)
        axs[1].set_xlabel("Timesteps")
        axs[1].legend()
        axs[1].set_xlim(0, timesteps.size)
        axs[1].set_ylim(1.3*np.min(adaptation_mock),
                        1.3*np.max(adaptation_mock))
        axs[1].set_title("Adaptation trace")
        fig.suptitle("Traces subjected to " + hardware_specific_effect)
        plt.savefig(self.plot_path.joinpath(path))
        plt.close()


if __name__ == "__main__":
    unittest.main()
