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
    cuba_li_integration, exp_cuba_li_integration)


class TestLIIntegration(unittest.TestCase):
    """ Test LI integration methods """

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_cuba_li_integration(self):
        """ Test CUBA LI integration """
        # Params
        tau_mem=6e-6
        tau_syn=6e-6
        leak=0

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        membrane_cadc, current, membrane_madc = cuba_li_integration(
            graded_spikes, leak=leak, tau_syn=tau_syn, tau_mem=tau_mem,
            dt=1e-6)

        # Shapes
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
        loss = membrane_cadc.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            membrane_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            current[:, 0].detach().numpy())

        plt.savefig(self.plot_path.joinpath("./cuba_li_dynamics_mock.png"))

    def test_cuba_li_integration_hw_data(self):
        """ Test CUBA LI integration with hardware data """
        # Params
        tau_mem=6e-6
        tau_syn=6e-6
        leak=0

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        membrane_cadc, current, membrane_madc = cuba_li_integration(
            graded_spikes, leak=leak, tau_syn=tau_syn, tau_mem=tau_mem,
            dt=1e-6)

        # Add jitter
        membrane_cadc += torch.rand(membrane_cadc.shape) * 0.05

        # Inject
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        membrane_cadc_hw, current_hw, membrane_madc_hw = cuba_li_integration(
            graded_spikes, leak=leak, tau_syn=tau_syn, tau_mem=tau_mem,
            hw_data=(membrane_cadc, membrane_cadc), dt=1e-6)

        # Shapes
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
            torch.equal(membrane_cadc_hw, membrane_cadc_hw))

        # No error
        loss = membrane_cadc_hw.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            membrane_cadc_hw[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            current_hw[:, 0].detach().numpy())
        plt.savefig(self.plot_path.joinpath("./cuba_lif_dynamics_hw.png"))

    def test_exp_li_integration(self):
        """ Test CUBA LI integration """
        dt = 1e-6
        # Params
        tau_mem = np.exp(dt/6e-6)
        tau_syn = np.exp(dt/6e-6)
        leak = 0

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        membrane_cadc, current, membrane_madc = exp_cuba_li_integration(
            graded_spikes, leak=leak, tau_syn_exp=tau_syn, tau_mem_exp=tau_mem)

        # Shapes
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
        loss = membrane_cadc.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., dt * 100, dt),
            membrane_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., dt * 100, dt),
            current[:, 0].detach().numpy())

        plt.savefig(self.plot_path.joinpath("./exp_li_dynamics_mock.png"))

    def test_cuba_li_integration_hw_data(self):
        """ Test CUBA LI integration with hardware data """
        dt = 1e-6
        # Params
        tau_mem = np.exp(dt/6e-6)
        tau_syn = np.exp(dt/6e-6)
        leak = 0

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        membrane_cadc, current, membrane_madc = exp_cuba_li_integration(
            graded_spikes, leak=leak, tau_syn_exp=tau_syn, tau_mem_exp=tau_mem)

        # Add jitter
        membrane_cadc += torch.rand(membrane_cadc.shape) * 0.05

        # Inject
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        membrane_cadc_hw, current_hw, membrane_madc_hw = exp_cuba_li_integration(
            graded_spikes, leak=leak, tau_syn_exp=tau_syn, tau_mem_exp=tau_mem,
            hw_data=(membrane_cadc, membrane_cadc))

        # Shapes
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
            torch.equal(membrane_cadc_hw, membrane_cadc_hw))

        # No error
        loss = membrane_cadc_hw.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., dt * 100, dt),
            membrane_cadc_hw[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., dt * 100, dt),
            current_hw[:, 0].detach().numpy())
        plt.savefig(self.plot_path.joinpath("./exp_li_dynamics_hw.png"))


if __name__ == "__main__":
    unittest.main()
