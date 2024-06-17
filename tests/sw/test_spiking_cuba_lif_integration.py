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
    CUBALIFParams, cuba_lif_integration, cuba_refractory_lif_integration)


class TestLIFIntegration(unittest.TestCase):
    """ Test LIF integration methods """

    plot_path = Path(__file__).parent.joinpath("plots")

    def setUp(self):
        self.plot_path.mkdir(exist_ok=True)

    def test_cuba_lif_integration(self):
        """ Test CUBA LIF integration """
        # Params
        params = CUBALIFParams(
            tau_mem=6e-6,
            tau_syn=6e-6,
            refractory_time=1e-6,
            threshold=0.7,
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
        h_out = cuba_lif_integration(graded_spikes, params, dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out.spikes.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out.v_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out.current.shape)))
        self.assertIsNone(h_out.v_madc)

        # No error
        loss = h_out.spikes.sum()
        loss.backward()

        # plot
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out.v_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out.current[:, 0].detach().numpy())

        plt.savefig(self.plot_path.joinpath("./cuba_lif_dynamics_mock.png"))

    def test_cuba_lif_integration_hw_data(self):
        """ Test CUBA LIF integration with hardware data """
        # Params
        params = CUBALIFParams(
            tau_mem=6e-6,
            tau_syn=6e-6,
            refractory_time=1e-6,
            threshold=0.7,
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
        h_out = cuba_lif_integration(graded_spikes, params, dt=1e-6)

        # Add jitter
        h_out.v_cadc += torch.rand(h_out.v_cadc.shape) * 0.05
        h_out.spikes[
            torch.randint(100, (1,)), torch.randint(10, (1,)),
            torch.randint(15, (1,))] = 1

        # Inject
        graded_spikes = hxsnn.SynapseHandle(
            torch.nn.functional.linear(inputs, weight))
        h_out_hw = cuba_lif_integration(
            graded_spikes, params,
            hw_data=(h_out.spikes, h_out.v_cadc, h_out.v_cadc), dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out_hw.spikes.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out_hw.v_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out_hw.current.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out_hw.v_madc.shape)))

        # Check HW data is still the same
        self.assertTrue(torch.equal(h_out_hw.v_cadc, h_out.v_cadc))
        self.assertTrue(torch.equal(h_out_hw.spikes, h_out.spikes))

        # No error
        loss = h_out_hw.spikes.sum()
        loss.backward()

        # plot
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out_hw.v_cadc[:, 0].detach().numpy())
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out_hw.current[:, 0].detach().numpy())
        plt.savefig(self.plot_path.joinpath("./cuba_lif_dynamics_hw.png"))

    def test_refractory_lif_integration(self):
        """ Test refractory LIF integration """
        # Params
        params = CUBALIFParams(
            tau_mem=6e-6,
            tau_syn=6e-6,
            refractory_time=1e-6,
            threshold=0.7,
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
        h_out = cuba_refractory_lif_integration(graded_spikes, params, dt=1e-6)

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(h_out.spikes.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(h_out.v_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]), torch.tensor(h_out.current.shape)))
        self.assertIsNone(h_out.v_madc)

        # No error
        loss = h_out.spikes.sum()
        loss.backward()

        # plot
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out.v_cadc[:, 0].detach().numpy())
        plt.savefig(
            self.plot_path.joinpath("./cuba_refractory_lif_dynamic.png"))

    def test_refractory_lif_integration_hw_data(self):
        """ Test refractory LIF integration with hardware data """

        # Params
        params = CUBALIFParams(
            tau_mem=6e-6,
            tau_syn=6e-6,
            refractory_time=1e-6,
            threshold=0.7,
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
        h_out = cuba_refractory_lif_integration(graded_spikes, params)

        # Add jitter
        h_out.v_cadc += torch.rand(h_out.v_cadc.shape) * 0.05
        h_out.spikes[
            torch.randint(100, (1,)), torch.randint(10, (1,)),
            torch.randint(15, (1,))] = 1

        # Inject
        graded_spikes = hxsnn.SynapseHandle(
            torch.nn.functional.linear(inputs, weight))
        h_out_hw = \
            cuba_refractory_lif_integration(
                graded_spikes, params,
                hw_data=(h_out.spikes, h_out.v_cadc, h_out.v_cadc))

        # Shapes
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out_hw.spikes.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out_hw.v_cadc.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out_hw.current.shape)))
        self.assertTrue(
            torch.equal(
                torch.tensor([100, 10, 15]),
                torch.tensor(h_out_hw.v_madc.shape)))

        # Check HW data is still the same
        self.assertTrue(torch.equal(h_out_hw.spikes, h_out.spikes))
        self.assertTrue(torch.equal(h_out_hw.v_cadc, h_out.v_cadc))
        self.assertTrue(torch.equal(h_out_hw.v_madc, h_out.v_cadc))
        self.assertTrue(torch.equal(h_out_hw.current, h_out.current))

        # No error
        loss = h_out_hw.spikes.sum()
        loss.backward()

        # plot
        _, ax = plt.subplots()
        ax.plot(
            np.arange(0., 1e-6 * 100, 1e-6),
            h_out_hw.v_cadc[:, 0].detach().numpy())
        plt.savefig(
            self.plot_path.joinpath("./cuba_refractory_lif_dynamic_hw.png"))

if __name__ == "__main__":
    unittest.main()
