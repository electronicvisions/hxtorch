"""
Test CUBA-LIF integration function
"""
import unittest
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import hxtorch.snn as hxsnn
from hxtorch.spiking.functional import CUBALIParams, cuba_li_integration


class TestLIIntegration(unittest.TestCase):
    """ Test LI integration methods """

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_cuba_li_integration(self):
        """ Test CUBA LI integration """
        # Params
        params = CUBALIParams(tau_mem=6e-6, tau_syn=6e-6)

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = hxsnn.SynapseHandle(
            torch.nn.functional.linear(inputs, weight))
        h_out = cuba_li_integration(graded_spikes, params, dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out.membrane_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(h_out.current.shape)))
        self.assertIsNone(h_out.membrane_madc)

        # No error
        loss = h_out.membrane_cadc.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out.membrane_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out.current[:, 0].detach().numpy())

        plt.savefig(self.plot_path.joinpath("./cuba_li_dynamics_mock.png"))

    def test_cuba_li_integration_hw_data(self):
        """ Test CUBA LI integration with hardware data """
        # Params
        params = CUBALIParams(tau_mem=6e-6, tau_syn=6e-6)

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = hxsnn.SynapseHandle(
            torch.nn.functional.linear(inputs, weight))
        h_out = cuba_li_integration(graded_spikes, params, dt=1e-6)

        # Add jitter
        h_out.membrane_cadc += torch.rand(h_out.membrane_cadc.shape) * 0.05

        # Inject
        graded_spikes = hxsnn.SynapseHandle(
            torch.nn.functional.linear(inputs, weight))
        h_out_hw = cuba_li_integration(
            graded_spikes, params,
            hw_data=(h_out.membrane_cadc, h_out.membrane_cadc))

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out_hw.membrane_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out_hw.current.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out_hw.membrane_madc.shape)))

        # Check HW data is still the same
        self.assertTrue(
            torch.equal(h_out_hw.membrane_cadc, h_out.membrane_cadc))

        # No error
        loss = h_out_hw.membrane_cadc.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out_hw.membrane_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out_hw.current[:, 0].detach().numpy())
        plt.savefig(self.plot_path.joinpath("./cuba_lif_dynamics_hw.png"))


if __name__ == "__main__":
    unittest.main()
