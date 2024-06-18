"""
Test Integrate and Fire integration
"""
import unittest
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

import hxtorch.snn as hxsnn
from hxtorch.spiking.functional import (
    cuba_iaf_integration,
    cuba_refractory_iaf_integration)


class TestIAFIntegration(unittest.TestCase):
    """ Test IAF integration methods """

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_iaf_integration(self):
        """ Test IAF integration """
        # Params
        tau_mem = 6e-6
        tau_syn = 6e-6
        threshold = 1.
        reset = -0.1
        alpha = 50.0
        method = "superspike"

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane_cadc, current, membrane_madc = cuba_iaf_integration(
            graded_spikes, reset=reset, threshold=threshold, tau_syn=tau_syn,
            tau_mem=tau_mem, method=method, alpha=alpha, dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(spikes.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(membrane_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(current.shape)))
        self.assertIsNone(membrane_madc)

        # No error
        loss = spikes.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            membrane_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            current[:, 0].detach().numpy())
        plt.savefig(self.plot_path.joinpath("./cuba_iaf_dynamics.png"))

    def test_iaf_integration_hw_data(self):
        """ Test IAF integration with hardware data """
        # Params
        tau_mem = 6e-6
        tau_syn = 6e-6
        threshold = 1.
        reset = -0.1
        alpha = 50.0
        method = "superspike"

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane_cadc, current, membrane_madc = cuba_iaf_integration(
            graded_spikes, reset=reset, threshold=threshold, tau_syn=tau_syn,
            tau_mem=tau_mem, method=method, alpha=alpha,dt=1e-6)
        self.assertIsNone(membrane_madc)

        # Add jitter
        membrane_cadc += torch.rand(membrane_cadc.shape) * 0.05
        spikes[
            torch.randint(100, (1,)), torch.randint(10, (1,)),
            torch.randint(15, (1,))] = 1

        # Inject
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes_hw, membrane_cadc_hw, current_hw, membrane_madc_hw = \
            cuba_iaf_integration(
                graded_spikes, reset=reset, threshold=threshold,
                tau_syn=tau_syn, tau_mem=tau_mem, method=method, alpha=alpha,
                hw_data=(spikes, membrane_cadc, membrane_cadc), dt=1e-6)

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
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            membrane_cadc_hw[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            current_hw[:, 0].detach().numpy())
        plt.savefig(self.plot_path.joinpath("./cuba_iaf_dynamics_hw.png"))

    def test_refractory_iaf_integration(self):
        """ Test refractory IAF integration """
        # Params
        tau_mem = 6e-6
        tau_syn = 6e-6
        tau_ref = 1e-6
        threshold = 1.
        reset = -0.1
        alpha = 50.0
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
            cuba_refractory_iaf_integration(
                graded_spikes, reset=reset, threshold=threshold,
                tau_syn=tau_syn, tau_mem=tau_mem, refractory_time=tau_ref,
                method=method, alpha=alpha, dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(spikes.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(membrane_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(current.shape)))
        self.assertIsNone(membrane_madc)

        # No error
        loss = spikes.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            membrane_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            current[:, 0].detach().numpy())
        plt.savefig(
            self.plot_path.joinpath("./cuba_refractory_iaf_dynamics.png"))

    @unittest.skip("Refractory update after integration overwrites hw data")
    def test_refractory_iaf_integration_hw_data(self):
        """ Test refractory IAF integration with hardware data """
        # Params
        tau_mem = 6e-6
        tau_syn = 6e-6
        tau_ref = 1e-6
        threshold = 1.
        reset = -0.1
        alpha = 50.0
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
            cuba_refractory_iaf_integration(
                graded_spikes, reset=reset, threshold=threshold,
                tau_syn=tau_syn, tau_mem=tau_mem, tau_ref=tau_ref,
                method=method, alpha=alpha, dt=1e-6)
        self.assertIsNone(membrane_madc)

        # Add jitter
        membrane_cadc += torch.rand(membrane_cadc.shape) * 0.05
        spikes[
            torch.randint(100, (1,)), torch.randint(10, (1,)),
            torch.randint(15, (1,))] = 1

        # Inject
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes_hw, membrane_cadc_hw, current_hw, membrane_madc_hw = \
            cuba_refractory_iaf_integration(
                graded_spikes, reset=reset, threshold=threshold,
                tau_syn=tau_syn, tau_mem=tau_mem, tau_ref=tau_ref,
                method=method, alpha=alpha, hw_data=(spikes, membrane, None),
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
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            membrane_cadc_hw[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            current_hw[:, 0].detach().numpy())
        plt.savefig(
            self.plot_path.joinpath("./cuba_refractory_iaf_dynamics_hw.png"))


if __name__ == "__main__":
    unittest.main()
