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
        graded_spikes = hxsnn.SynapseHandle(
            torch.nn.functional.linear(inputs, weight))
        h_out = cuba_iaf_integration(graded_spikes, params, dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(h_out.spikes.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(h_out.membrane_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(h_out.current.shape)))
        self.assertIsNone(h_out.membrane_madc)

        # No error
        loss = h_out.spikes.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out.membrane_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out.current[:, 0].detach().numpy())
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
        graded_spikes = hxsnn.SynapseHandle(
            torch.nn.functional.linear(inputs, weight))
        h_out = cuba_iaf_integration(graded_spikes, params, dt=1e-6)
        self.assertIsNone(h_out.membrane_madc)

        # Add jitter
        h_out.membrane_cadc += torch.rand(h_out.membrane_cadc.shape) * 0.05
        h_out.spikes[
            torch.randint(100, (1,)), torch.randint(10, (1,)),
            torch.randint(15, (1,))] = 1

        # Inject
        graded_spikes = hxsnn.SynapseHandle(
            torch.nn.functional.linear(inputs, weight))
        h_out_hw = cuba_iaf_integration(
            graded_spikes, params,
            hw_data=(h_out.spikes, h_out.membrane_cadc, h_out.membrane_cadc),
            dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out_hw.spikes.shape)))
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
        self.assertTrue(torch.equal(h_out_hw.spikes, h_out.spikes))

        # No error
        loss = h_out_hw.spikes.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out_hw.membrane_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out_hw.current[:, 0].detach().numpy())
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
        graded_spikes = hxsnn.SynapseHandle(
            torch.nn.functional.linear(inputs, weight))
        h_out = cuba_refractory_iaf_integration(graded_spikes, params, dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(h_out.spikes.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(h_out.membrane_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(h_out.current.shape)))
        self.assertIsNone(h_out.membrane_madc)

        # No error
        loss = h_out.spikes.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out.membrane_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out.current[:, 0].detach().numpy())
        plt.savefig(
            self.plot_path.joinpath("./cuba_refractory_iaf_dynamics.png"))

    def test_refractory_iaf_integration_hw_data(self):
        """ Test refractory IAF integration with hardware data """
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
        graded_spikes = hxsnn.SynapseHandle(
            torch.nn.functional.linear(inputs, weight))
        h_out = cuba_refractory_iaf_integration(graded_spikes, params, dt=1e-6)
        self.assertIsNone(h_out.membrane_madc)

        # Add jitter
        h_out.membrane_cadc += torch.rand(h_out.membrane_cadc.shape) * 0.05
        h_out.spikes[
            torch.randint(100, (1,)), torch.randint(10, (1,)),
            torch.randint(15, (1,))] = 1

        # Inject
        graded_spikes = hxsnn.SynapseHandle(
            torch.nn.functional.linear(inputs, weight))
        h_out_hw = \
            cuba_refractory_iaf_integration(
                graded_spikes, params,
                hw_data=(
                      h_out.spikes, h_out.membrane_cadc, h_out.membrane_cadc),
                dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out_hw.spikes.shape)))
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
        self.assertTrue(torch.equal(h_out_hw.spikes, h_out.spikes))

        # No error
        loss = h_out_hw.spikes.sum()
        loss.backward()

        # plot
        fig, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out_hw.membrane_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out_hw.current[:, 0].detach().numpy())
        plt.savefig(
            self.plot_path.joinpath("./cuba_refractory_iaf_dynamics_hw.png"))


if __name__ == "__main__":
    unittest.main()
