"""
Test Integrate and Fire integration
"""
import unittest
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from hxtorch.spiking.functional import (
    CUBAIAFParams, cuba_iaf_integration,
    cuba_refractory_iaf_integration)


class TestIAFIntegration(unittest.TestCase):
    """ Test IAF integration methods """

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_iaf_integration(self):
        """ Test IAF integration """
        # Params
        params = CUBAIAFParams(
            tau_mem=6e-6,
            tau_syn=6e-6,
            refractory_time=1e-6,
            threshold=1.,
            reset=-0.1)

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane, current, v_madc = cuba_iaf_integration(
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
        self.assertIsNone(v_madc)

        # No error
        loss = spikes.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), membrane[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), current[:, 0].detach().numpy())
        plt.savefig(self.plot_path.joinpath("./cuba_iaf_dynamics.png"))

    def test_iaf_integration_hw_data(self):
        """ Test IAF integration with hardware data """
        # Params
        params = CUBAIAFParams(
            tau_mem=6e-6,
            tau_syn=6e-6,
            refractory_time=0e-6,
            threshold=1.,
            reset=-0.1)

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane, current, v_madc = cuba_iaf_integration(
            graded_spikes, params, dt=1e-6)
        self.assertIsNone(v_madc)

        # Add jitter
        membrane += torch.rand(membrane.shape) * 0.05
        spikes[
            torch.randint(100, (1,)), torch.randint(10, (1,)),
            torch.randint(15, (1,))] = 1

        # Inject
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes_hw, membrane_hw, current, v_madc = cuba_iaf_integration(
            graded_spikes, params, hw_data=(spikes, membrane, membrane),
            dt=1e-6)

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
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(v_madc.shape)))

        # Check HW data is still the same
        self.assertTrue(torch.equal(membrane_hw, membrane))
        self.assertTrue(torch.equal(spikes_hw, spikes))

        # No error
        loss = spikes.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), membrane[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), current[:, 0].detach().numpy())
        plt.savefig(self.plot_path.joinpath("./cuba_iaf_dynamics_hw.png"))

    def test_refractory_iaf_integration(self):
        """ Test refractory IAF integration """
        # Params
        params = CUBAIAFParams(
            tau_mem=6e-6,
            tau_syn=6e-6,
            refractory_time=1e-6,
            threshold=1.,
            reset=-0.1)

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane, current, v_madc = cuba_refractory_iaf_integration(
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
        self.assertIsNone(v_madc)

        # No error
        loss = spikes.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), membrane[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), current[:, 0].detach().numpy())
        plt.savefig(
            self.plot_path.joinpath("./cuba_refractory_iaf_dynamics.png"))

    def test_refractory_iaf_integration_hw_data(self):
        """ Test refractory IAF integration with hardware data """
        # Params
        params = CUBAIAFParams(
            tau_mem=6e-6,
            tau_syn=6e-6,
            refractory_time=0e-6,
            threshold=1.,
            reset=-0.1)

        # Inputs
        inputs = torch.zeros(100, 10, 5)
        inputs[10, :, 0] = 1
        inputs[30, :, 1] = 1
        inputs[40, :, 2] = 1
        inputs[53, :, 3] = 1

        weight = torch.nn.parameter.Parameter(torch.randn(15, 5))
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes, membrane, current, v_madc = cuba_refractory_iaf_integration(
            graded_spikes, params, dt=1e-6)
        self.assertIsNone(v_madc)

        # Add jitter
        membrane += torch.rand(membrane.shape) * 0.05
        spikes[
            torch.randint(100, (1,)), torch.randint(10, (1,)),
            torch.randint(15, (1,))] = 1

        # Inject
        graded_spikes = torch.nn.functional.linear(inputs, weight)
        spikes_hw, membrane_hw, current, v_madc = \
            cuba_refractory_iaf_integration(
                graded_spikes, params, hw_data=(spikes, membrane, membrane),
                dt=1e-6)

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
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(v_madc.shape)))

        # Check HW data is still the same
        self.assertTrue(torch.equal(membrane_hw, membrane))
        self.assertTrue(torch.equal(spikes_hw, spikes))

        # No error
        loss = spikes.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), membrane[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6), current[:, 0].detach().numpy())
        plt.savefig(
            self.plot_path.joinpath("./cuba_refractory_iaf_dynamics_hw.png"))


if __name__ == "__main__":
    unittest.main()
