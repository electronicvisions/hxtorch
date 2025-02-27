"""
Test CUBA-LIF integration function
"""
import unittest
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import hxtorch.snn as hxsnn
from hxtorch.spiking.functional import (
    cuba_lif_integration, cuba_refractory_lif_integration,
    exp_cuba_lif_integration)


class TestLIFIntegration(unittest.TestCase):
    """ Test LIF integration methods """

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_cuba_lif_integration(self):
        """ Test CUBA LIF integration """

        # Params
        tau_mem = 6e-6
        tau_syn = 6e-6
        threshold = 0.7
        reset = -0.1
        leak = 0
        alpha = 50
        method = "superspike"

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane_cadc, current, membrane_madc = cuba_lif_integration(
            graded_spikes, leak=leak, reset=reset, tau_mem=tau_mem,
            tau_syn=tau_syn, threshold=threshold, method=method, alpha=alpha,
            dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(spikes.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(membrane_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(current.shape)))
        self.assertIsNone(membrane_madc)

        # No error
        loss = spikes.sum()
        loss.backward()

        # plot
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            membrane_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            current[:, 0].detach().numpy())

        plt.savefig(self.plot_path.joinpath("./cuba_lif_dynamics_mock.png"))

    def test_cuba_lif_integration_hw_data(self):
        """ Test CUBA LIF integration with hardware data """
        # Params
        tau_mem = 6e-6
        tau_syn = 6e-6
        threshold = 0.7
        reset = -0.1
        leak = 0
        alpha = 50
        method = "superspike"

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane_cadc, current, membrane_madc = cuba_lif_integration(
            graded_spikes, leak=leak, reset=reset, tau_mem=tau_mem,
            tau_syn=tau_syn, threshold=threshold, method=method, alpha=alpha,
            dt=1e-6)
        self.assertIsNone(membrane_madc)

        # Add jitter
        membrane_cadc += torch.rand(membrane_cadc.shape) * 0.05
        spikes[
            torch.randint(100, (1,)), torch.randint(10, (1,)),
            torch.randint(15, (1,))] = 1

        # Inject
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes_hw, membrane_cadc_hw, current_hw, membrane_madc_hw = \
            cuba_lif_integration(
                graded_spikes, leak=leak, reset=reset, tau_mem=tau_mem,
                tau_syn=tau_syn, threshold=threshold, method=method,
                alpha=alpha, hw_data=(spikes, membrane_cadc, membrane_cadc),
                dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(spikes_hw.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(membrane_cadc_hw.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(current_hw.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(membrane_madc_hw.shape)))

        # Check HW data is still the same
        self.assertTrue(
            torch.equal(membrane_cadc_hw, membrane_cadc))
        self.assertTrue(torch.equal(spikes_hw, spikes))

        # No error
        loss = spikes_hw.sum()
        loss.backward()

        # plot
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            membrane_cadc_hw[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            current_hw[:, 0].detach().numpy())
        plt.savefig(self.plot_path.joinpath("./cuba_lif_dynamics_hw.png"))

    def test_refractory_lif_integration(self):
        """ Test refractory LIF integration """
        # Params
        leak = 0
        tau_mem = 6e-6
        tau_syn = 6e-6
        refractory_time = 1e-6
        threshold = 0.7
        reset = -0.1
        alpha = 50
        method = "superspike"

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane_cadc, current, membrane_madc = \
            cuba_refractory_lif_integration(
                graded_spikes, leak=leak, reset=reset, tau_mem=tau_mem,
                tau_syn=tau_syn, threshold=threshold, alpha=alpha,
                method=method, refractory_time=refractory_time, dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(spikes.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(membrane_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(current.shape)))
        self.assertIsNone(membrane_madc)

        # No error
        loss = spikes.sum()
        loss.backward()

        # plot
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            membrane_cadc[:, 0].detach().numpy())
        plt.savefig(
            self.plot_path.joinpath("./cuba_refractory_lif_dynamic.png"))

    def test_refractory_lif_integration_hw_data(self):
        """ Test refractory LIF integration with hardware data """

        # Params
        leak = 0
        tau_mem = 6e-6
        tau_syn = 6e-6
        refractory_time = 1e-6
        threshold = 0.7
        reset = -0.1
        alpha = 50
        method = "superspike"

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane_cadc, current, membrane_madc = \
            cuba_refractory_lif_integration(
                graded_spikes, leak=leak, reset=reset, tau_mem=tau_mem,
                tau_syn=tau_syn, threshold=threshold, method=method,
                alpha=alpha, refractory_time=refractory_time, dt=1e-6)

        # Add jitter
        membrane_cadc += torch.rand(membrane_cadc.shape) * 0.05
        spikes[
            torch.randint(100, (1,)), torch.randint(10, (1,)),
            torch.randint(15, (1,))] = 1

        # Inject
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes_hw, membrane_cadc_hw, current_hw, membrane_madc_hw = \
            cuba_refractory_lif_integration(
                graded_spikes, leak=leak, reset=reset, tau_mem=tau_mem,
                tau_syn=tau_syn, threshold=threshold,
                refractory_time=refractory_time, alpha=alpha, method=method,
                hw_data=(spikes, membrane_cadc, membrane_cadc))

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(spikes_hw.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(membrane_cadc_hw.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(current_hw.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(membrane_madc_hw.shape)))

        # Check HW data is still the same
        self.assertTrue(torch.equal(spikes_hw, spikes))
        self.assertTrue(torch.equal(membrane_cadc_hw, membrane_cadc))
        self.assertTrue(torch.equal(membrane_madc_hw, membrane_cadc))
        self.assertTrue(torch.equal(current_hw, current))

        # No error
        loss = spikes_hw.sum()
        loss.backward()

        # plot
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            membrane_cadc_hw[:, 0].detach().numpy())
        plt.savefig(
            self.plot_path.joinpath("./cuba_refractory_lif_dynamic_hw.png"))

    def test_cuba_lif_exp_integration(self):
        """ Test CUBA LIF integration with exponentials """

        dt = 1e-6
        # Params
        tau_mem = np.exp(dt/6e-6)
        tau_syn = np.exp(dt/6e-6)
        threshold = 0.7
        reset = -0.1
        leak = 0
        alpha = 50
        method = "superspike"

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane_cadc, current, membrane_madc = exp_cuba_lif_integration(
            graded_spikes, leak=leak, reset=reset, tau_mem_exp=tau_mem,
            tau_syn_exp=tau_syn, threshold=threshold, method=method, alpha=alpha)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(spikes.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(membrane_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(current.shape)))
        self.assertIsNone(membrane_madc)

        # No error
        loss = spikes.sum()
        loss.backward()

        # plot
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., dt * 100, dt),
            membrane_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., dt * 100, dt),
            current[:, 0].detach().numpy())

        plt.savefig(self.plot_path.joinpath("./cuba_lif_dynamics_mock.png"))

    def test_cuba_lif_integration_hw_data(self):
        """ Test CUBA LIF integration with hardware data """
        dt = 1e-6
        # Params
        tau_mem = np.exp(dt/6e-6)
        tau_syn = np.exp(dt/6e-6)
        threshold = 0.7
        reset = -0.1
        leak = 0
        alpha = 50
        method = "superspike"

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane_cadc, current, membrane_madc = exp_cuba_lif_integration(
            graded_spikes, leak=leak, reset=reset, tau_mem_exp=tau_mem,
            tau_syn_exp=tau_syn, threshold=threshold, method=method, alpha=alpha)
        self.assertIsNone(membrane_madc)

        # Add jitter
        membrane_cadc += torch.rand(membrane_cadc.shape) * 0.05
        spikes[
            torch.randint(100, (1,)), torch.randint(10, (1,)),
            torch.randint(15, (1,))] = 1

        # Inject
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes_hw, membrane_cadc_hw, current_hw, membrane_madc_hw = \
            exp_cuba_lif_integration(
                graded_spikes, leak=leak, reset=reset, tau_mem_exp=tau_mem,
                tau_syn_exp=tau_syn, threshold=threshold, method=method,
                alpha=alpha, hw_data=(spikes, membrane_cadc, membrane_cadc))

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(spikes_hw.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(membrane_cadc_hw.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(current_hw.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(membrane_madc_hw.shape)))

        # Check HW data is still the same
        self.assertTrue(
            torch.equal(membrane_cadc_hw, membrane_cadc))
        self.assertTrue(torch.equal(spikes_hw, spikes))

        # No error
        loss = spikes_hw.sum()
        loss.backward()

        # plot
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., dt * 100, dt),
            membrane_cadc_hw[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., dt * 100, dt),
            current_hw[:, 0].detach().numpy())
        plt.savefig(self.plot_path.joinpath("./cuba_lif_dynamics_hw.png"))

if __name__ == "__main__":
    unittest.main()
