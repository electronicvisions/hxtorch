"""
Test CUBA-LIF integration function
"""
import unittest
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from hxtorch.spiking.functional import CUBALIParams, cuba_li_integration


class TestLIIntegration(unittest.TestCase):
    """ Test LI integration methods """

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_cuba_li_integration(self):
        """ Test CUBA LI integration """
        # Params
        params = CUBALIParams(tau_mem_inv=1./6e-6, tau_syn_inv=1./6e-6)

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        membrane, current, v_madc = cuba_li_integration(
            graded_spikes, params, dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(membrane.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(current.shape)))
        self.assertIsNone(v_madc)

        # No error
        loss = membrane.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), membrane[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), current[:, 0].detach().numpy())

        plt.savefig(self.plot_path.joinpath("./cuba_li_dynamics_mock.png"))

    def test_cuba_li_integration_hw_data(self):
        """ Test CUBA LI integration with hardware data """
        # Params
        params = CUBALIParams(tau_mem_inv=1./6e-6, tau_syn_inv=1./6e-6)

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        membrane, current, v_madc = cuba_li_integration(
            graded_spikes, params, dt=1e-6)

        # Add jitter
        membrane += torch.rand(membrane.shape) * 0.05

        # Inject
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        membrane_hw, current, v_madc = cuba_li_integration(
            graded_spikes, params, hw_data=(membrane, membrane), dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(membrane.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(current.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(v_madc.shape)))

        # Check HW data is still the same
        self.assertTrue(torch.equal(membrane_hw, membrane))

        # No error
        loss = membrane_hw.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), membrane[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), current[:, 0].detach().numpy())
        plt.savefig(self.plot_path.joinpath("./cuba_lif_dynamics_hw.png"))


if __name__ == "__main__":
    unittest.main()
