"""
Test mocking of fixed pattern noise via MockParameter
"""
import unittest
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt

from hxtorch.core.parameter import ModelParameter, MockParameter
import hxtorch.spiking as hxsnn


# pylint: disable=too-many-instance-attributes
class TestMockParameter(unittest.TestCase):
    """
    Test mocking of fixed pattern noise via MockParameter
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
        self.leak = 0.
        self.reset = 0.
        self.threshold = 1.
        self.g_l = 1.
        self.refractory_time = 1e-6
        self.exp_slope = 0.2
        self.exp_threshold = 0.3
        self.spike_triggered_adaptation_increment = 0.2
        # Set random seed for reproducability
        torch.manual_seed(42)

    def test_mock_parameter(self):
        """
        Test, if functionalities of MockParameter work, as intended.
        """
        # Test automatic sampling when (re)setting mean and std
        noisy_tau_mem = MockParameter(mean=10e-6, std=2e-6)
        copy_original = deepcopy(noisy_tau_mem.model_value)
        noisy_tau_mem.mean = 20e-6
        copy_different_mean = deepcopy(noisy_tau_mem.model_value)
        noisy_tau_mem.std = 5e-6
        copy_different_std = deepcopy(noisy_tau_mem.model_value)
        self.assertNotEqual(copy_original, 10e-6)
        self.assertNotEqual(copy_original, copy_different_mean)
        self.assertNotEqual(copy_original, copy_different_std)
        self.assertNotEqual(copy_different_std, copy_different_mean)

        # Test shaping of model_value
        noisy_tau_mem.sample(self.size)
        shaped_tau_mem_sample = noisy_tau_mem.model_value
        self.assertEqual(torch.Size((10,)), shaped_tau_mem_sample.shape)

        # Shaping with other size than mean or std not possible:
        faulty_tau_mem = MockParameter(mean=torch.ones(4)*10e-6,
                                       std=torch.ones(4)*2e-6)
        self.assertRaises(RuntimeError, faulty_tau_mem.sample, (self.size,))

        # Test eqivalence to ModelParameter if std is None
        model_param = ModelParameter(10e-6)
        mock_param = MockParameter(10e-6)
        self.assertEqual(model_param.model_value, mock_param.model_value)

    # pylint: disable=too-many-locals
    def test_fixed_pattern_noise(self):
        """
        Simulate population of neurons with fixed pattern noise on some
        parameters. Plot membrane and adaptation traces of population with
        noisy parameters, as well as from a neuron with non-varying parameters.
        """
        weights = torch.nn.parameter.Parameter(torch.ones(self.size, 10))
        graded_spikes = torch.nn.functional.linear(self.inputs, weights)

        # Set non-noisy parameters to simulate ideal equations:
        tau_syn_clean_sim = 10e-6
        tau_mem_clean_sim = 10e-6
        subthreshold_adaptation_strength_clean_sim = 2.
        tau_adap_clean_sim = 100e-6

        integration_step_code = hxsnn.functional.CuBaStepCode(
            leaky=True, fire=True, refractory=True, exponential=True,
            subthreshold_adaptation=True, spike_triggered_adaptation=True,
            hw_voltage_trace_available=False,
            hw_adaptation_trace_available=False, hw_spikes_available=False).\
            generate()
        (membrane_cadc_clean_sim, _, _, adaptation_cadc_clean_sim, _,
            spikes_clean_sim) = hxsnn.functional.cuba_aelif_integration(
                graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, c_mem=tau_mem_clean_sim,
                g_l=self.g_l, tau_syn=tau_syn_clean_sim,
                refractory_time=self.refractory_time, method="superspike",
                alpha=50, exp_slope=self.exp_slope,
                exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    subthreshold_adaptation_strength_clean_sim),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=tau_adap_clean_sim, dt=2e-6, leaky=True, fire=True,
                refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=integration_step_code)

        # Set noisy parameters subjected to fixed pattern noise:
        tau_syn_mock = MockParameter(mean=10e-6, std=2e-6)
        tau_syn_mock.sample(self.size)
        tau_mem_mock = MockParameter(mean=10e-6, std=2e-6)
        tau_mem_mock.sample(self.size)
        subthreshold_adaptation_strength_mock = MockParameter(
            mean=2., std=0.5)
        subthreshold_adaptation_strength_mock.sample(self.size)
        tau_adap_mock = MockParameter(mean=100e-6, std=20e-6)
        tau_adap_mock.sample(self.size)

        membrane_cadc_mock, _, _, adaptation_cadc_mock, _, spikes_mock = \
            hxsnn.functional.cuba_aelif_integration(
                graded_spikes, leak=self.leak, reset=self.reset,
                threshold=self.threshold, c_mem=tau_mem_mock.model_value,
                g_l=self.g_l, tau_syn=tau_syn_mock.model_value,
                refractory_time=self.refractory_time, method="superspike",
                alpha=50, exp_slope=self.exp_slope,
                exp_threshold=self.exp_threshold,
                subthreshold_adaptation_strength=(
                    subthreshold_adaptation_strength_mock.model_value),
                spike_triggered_adaptation_increment=(
                    self.spike_triggered_adaptation_increment),
                tau_adap=tau_adap_mock.model_value, dt=2e-6, leaky=True,
                fire=True, refractory=True, exponential=True,
                subthreshold_adaptation=True, spike_triggered_adaptation=True,
                integration_step_code=integration_step_code)

        # Plot traces for manual visual control
        self.plot("fixed_pattern_noise.png", membrane_cadc_clean_sim,
                  adaptation_cadc_clean_sim, spikes_clean_sim,
                  membrane_cadc_mock, adaptation_cadc_mock, spikes_mock)

    # pylint: disable=too-many-locals, too-many-arguments
    def plot(self, path, membrane_clean_sim, adaptation_clean_sim,
             spikes_clean_sim, membrane_mock, adaptation_mock, spikes_mock):
        """
        Plot voltage traces and spikes, as well as adaptation traces of both
        the simulation with and without fixed patter noise.
        """
        membrane_sim = membrane_clean_sim[:, 0, 0].detach().numpy()
        membranes_mock = membrane_mock[:, 0, :].detach().numpy()
        adaptation_sim = adaptation_clean_sim[:, 0, 0].detach().numpy()
        adaptations_mock = adaptation_mock[:, 0, :].detach().numpy()
        spike_indices_sim = spikes_clean_sim.detach()[:, 0, 0].nonzero()[:, 0]
        spike_indices_mock = spikes_mock.detach()[:, 0, :].nonzero()[:, 0]
        timesteps = np.arange(0, membranes_mock.shape[0], 1)
        fig = plt.figure(layout="tight", figsize=(8, 8))
        axs = fig.subplots(2, 1, sharex=False)
        axs[0].plot(timesteps, membranes_mock,
                    alpha=0.5, linewidth=0.5, color='k')
        axs[0].vlines(spike_indices_mock, ymin=-1e2, ymax=1e2,  color='red',
                      linewidth=0.5, alpha=0.5)
        axs[0].plot(timesteps, membrane_sim,
                    color='green', label="without fixed pattern noise")
        axs[0].vlines(spike_indices_sim, ymin=-1e2, ymax=1e2, color='red',
                      linewidth=1., label="spikes")
        axs[0].set_xlabel("Timesteps")
        axs[0].legend()
        axs[0].set_xlim(0, timesteps.size)
        axs[0].set_ylim(1.3*np.min(membranes_mock), 1.3*np.max(membranes_mock))
        axs[0].set_title("Membrane traces")
        axs[1].plot(timesteps, adaptations_mock,
                    alpha=0.5, linewidth=0.5, color='k')
        axs[1].plot(timesteps, adaptation_sim,
                    color='green', label="without fixed pattern noise")
        axs[1].set_xlabel("Timesteps")
        axs[1].legend()
        axs[1].set_xlim(0, timesteps.size)
        axs[1].set_ylim(1.3*np.min(adaptations_mock),
                        1.3*np.max(adaptations_mock))
        axs[1].set_title("Adaptation traces")
        fig.suptitle("Traces subjected to fixed pattern noise")
        plt.savefig(self.plot_path.joinpath(path))
        plt.close()


if __name__ == "__main__":
    unittest.main()
