"""
Test mocking of random noise such as temporal noise on the state variables
and ADC readout noise
"""
import unittest
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from hxtorch.spiking.functional.mock import RandomNoise
import hxtorch.spiking as hxsnn


# pylint: disable=too-many-instance-attributes
class TestRandomNoise(unittest.TestCase):
    """
    Test mocking of random noise
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
            noisy_traces=True).generate()
        # Set random seed for reproducability
        torch.manual_seed(42)

    def test_random_noise(self):
        """
        Test, if functionalities of RandomNoise work, as intended.
        """
        # Test if zeros are returned, if std is not specified / None
        noise = RandomNoise()
        sample = noise.sample()
        self.assertTrue(torch.equal(torch.tensor([0.]), sample))

        # Test for correct shapes of sampled values
        noise = RandomNoise(std=0.1)
        sample = noise.sample()
        self.assertEqual(sample.shape, torch.Size())
        sample = noise.sample(torch.Size((10, 10, 10)))
        self.assertEqual(sample.shape, torch.Size((10, 10, 10)))

        noise = RandomNoise(std=torch.ones((11, 11, 11)) * 0.1)
        sample = noise.sample()
        self.assertEqual(sample.shape, torch.Size((11, 11, 11)))
        self.assertRaises(RuntimeError, noise.sample, (10, 11, 11))

        # Test for auto-assignment to default device
        default_dev = torch.device("cpu")
        noise = RandomNoise(std=0.1)
        sample_1 = noise.sample()
        noise.to(default_dev)
        sample_2 = noise.sample()
        self.assertEqual(sample_1.device, sample_2.device)

        # Test subscription
        explicit_noise = RandomNoise(std=torch.linspace(0., 0.1, self.size))
        crop_idx = int(self.size / 2)
        sliced_noise = explicit_noise[:crop_idx]
        self.assertTrue(torch.allclose(
            sliced_noise.std, explicit_noise.std[:crop_idx]))

    def test_temporal_noise(self):
        """
        Simulate the trace of one neuron with and without noise on the
        voltage- and adaptation trace. Plot both of them in comparison and
        assert non-equality.
        """
        weights = torch.nn.parameter.Parameter(torch.ones(self.size, 10))
        graded_spikes = torch.nn.functional.linear(self.inputs, weights)

        # Generate traces without temporal noise
        (membrane_cadc_clean_sim, _, current_cadc_clean_sim,
            adaptation_cadc_clean_sim, _,
            spikes_clean_sim) = hxsnn.functional.cuba_aelif_integration(
                graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, c_mem=self.tau_mem,
                g_l=self.g_l, tau_syn=self.tau_syn,
                refractory_time=self.refractory_time, method="superspike",
                alpha=50, exp_slope=self.exp_slope,
                exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    self.subthreshold_adaptation_strength),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, dt=2e-6, leaky=True, fire=True,
                refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=self.integration_step_code)

        # Generate traces with temporal noise
        trace_noise_current = RandomNoise(0.)
        trace_noise_voltage = RandomNoise(torch.full([self.size], 0.01))
        trace_noise_adaptation = RandomNoise(0.02)
        (membrane_cadc_noisy_traces, _, current_cadc_noisy_traces,
            adaptation_cadc_noisy_traces, _,
            spikes_noisy_traces) = hxsnn.functional.cuba_aelif_integration(
                graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, c_mem=self.tau_mem,
                g_l=self.g_l, tau_syn=self.tau_syn,
                refractory_time=self.refractory_time, method="superspike",
                alpha=50, exp_slope=self.exp_slope,
                exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    self.subthreshold_adaptation_strength),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, dt=2e-6,
                trace_noise_current=trace_noise_current,
                trace_noise_voltage=trace_noise_voltage,
                trace_noise_adaptation=trace_noise_adaptation, leaky=True,
                fire=True, refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=self.integration_step_code)

        # Assert equailty or non-equality respectively
        self.assertTrue(torch.allclose(
            current_cadc_clean_sim, current_cadc_noisy_traces))
        self.assertFalse(torch.allclose(
            membrane_cadc_clean_sim, membrane_cadc_noisy_traces))
        self.assertFalse(torch.allclose(
            adaptation_cadc_clean_sim, adaptation_cadc_noisy_traces))

        # Plot differing traces for manual visual control
        self.plot("temporal_noise.png", membrane_cadc_clean_sim,
                  adaptation_cadc_clean_sim, spikes_clean_sim,
                  membrane_cadc_noisy_traces, adaptation_cadc_noisy_traces,
                  spikes_noisy_traces, "temporal noise")

    def test_readout_noise(self):
        """
        Simulate the trace of one neuron with and without cadc readout noise
        on the voltage- and adaptation trace. Plot both of them in comparison
        and assert non-equality.
        """
        weights = torch.nn.parameter.Parameter(torch.ones(self.size, 10))
        graded_spikes = torch.nn.functional.linear(self.inputs, weights)

        # Generate traces without temporal noise
        (membrane_cadc_clean_sim, _, current_cadc_clean_sim,
            adaptation_cadc_clean_sim, _,
            spikes_clean_sim) = hxsnn.functional.cuba_aelif_integration(
                graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, c_mem=self.tau_mem,
                g_l=self.g_l, tau_syn=self.tau_syn,
                refractory_time=self.refractory_time, method="superspike",
                alpha=50, exp_slope=self.exp_slope,
                exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=0.,
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, dt=2e-6, leaky=True, fire=True,
                refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=self.integration_step_code)

        # Generate traces with temporal noise
        cadc_readout_noise_voltage = RandomNoise(
            torch.linspace(0., 0.1, self.size))
        cadc_readout_noise_adaptation = RandomNoise(None)
        (membrane_cadc_noisy_traces, _, current_cadc_noisy_traces,
            adaptation_cadc_noisy_traces, _,
            spikes_noisy_traces) = hxsnn.functional.cuba_aelif_integration(
                graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, c_mem=self.tau_mem,
                g_l=self.g_l, tau_syn=self.tau_syn,
                refractory_time=self.refractory_time, method="superspike",
                alpha=50, exp_slope=self.exp_slope,
                exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=0.,
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=self.tau_adap, dt=2e-6,
                cadc_readout_noise_voltage=cadc_readout_noise_voltage,
                cadc_readout_noise_adaptation=cadc_readout_noise_adaptation,
                fire=True, refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=self.integration_step_code)

        # Assert equailty or non-equality respectively
        self.assertTrue(torch.allclose(
            current_cadc_clean_sim, current_cadc_noisy_traces))
        self.assertFalse(torch.allclose(
            membrane_cadc_clean_sim, membrane_cadc_noisy_traces))
        self.assertTrue(torch.allclose(
            adaptation_cadc_clean_sim, adaptation_cadc_noisy_traces))

        # Plot differing traces for manual visual control
        self.plot("readout_noise.png", membrane_cadc_clean_sim,
                  adaptation_cadc_clean_sim, spikes_clean_sim,
                  membrane_cadc_noisy_traces, adaptation_cadc_noisy_traces,
                  spikes_noisy_traces, "readout noise")

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
                      linewidth=1., alpha=0.5)
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
