"""
Test CUBA-LIF integration function
"""
import unittest
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from hxtorch.snn.functional import CUBALIFParams, cuba_lif_integration


class TestLIFIntegration(unittest.TestCase):
    """ Test LIF integration methods """

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_cuba_lif_integration(self):
        """ Test CUBA LIF integration """
        # Params
        params = CUBALIFParams(
            tau_mem_inv=1./6e-6,
            tau_syn_inv=1./6e-6,
            tau_ref=1e-6,
            v_th=0.7,
            v_reset=-0.1)

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane, current = cuba_lif_integration(
            graded_spikes, params, dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(spikes.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(membrane.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(current.shape)))

        # No error
        loss = spikes.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), membrane[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), current[:, 0].detach().numpy())

        plt.savefig(self.plot_path.joinpath("./cuba_lif_dynamics_mock.png"))

    def test_cuba_lif_integration_hw_data(self):
        """ Test CUBA LIF integration with hardware data """
        # Params
        params = CUBALIFParams(
            tau_mem_inv=1./6e-6,
            tau_syn_inv=1./6e-6,
            tau_ref=0e-6,
            v_th=1.,
            v_reset=-0.1)

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane, current = cuba_lif_integration(
            graded_spikes, params, dt=1e-6)

        # Add jitter
        membrane += torch.rand(membrane.shape) * 0.05
        spikes[
            torch.randint(100, (1,)), torch.randint(10, (1,)),
            torch.randint(15, (1,))] = 1

        # Inject
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes_hw, membrane_hw, current = cuba_lif_integration(
            graded_spikes, params, hw_data=(spikes, membrane), dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(spikes.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(membrane.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(current.shape)))

        # Check HW data is still the same
        self.assertTrue(torch.equal(membrane_hw, membrane))
        self.assertTrue(torch.equal(spikes_hw, spikes))

        # No error
        loss = spikes_hw.sum()
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
