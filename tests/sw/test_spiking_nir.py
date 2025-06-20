import unittest
import nir
import numpy as np
import torch

import hxtorch.spiking as hxsnn
from hxtorch.examples.spiking.yinyang_model import SNN
from hxtorch.spiking.utils.from_nir import (_map_nir_to_hxtorch, 
    ConversionConfig)
from hxtorch.spiking.utils.to_nir import _map_hxtorch_to_nir, to_nir


class TestToNIRConversion(unittest.TestCase):

    def test_cubali_to_nir(self):
        exp = hxsnn.Experiment(mock=True)
        neuron = hxsnn.LI(
            size=10,
            experiment=exp,
            leak=hxsnn.MixedHXModelParameter(0.1, 80),
            tau_mem=0.02,
            tau_syn=0.005,
            trace_scale=1.0,
            cadc_time_shift=0.001,
            shift_cadc_to_first=True,
            enable_cadc_recording=True
        )

        nir_neuron = _map_hxtorch_to_nir(neuron)

        self.assertIsInstance(nir_neuron, nir.CubaLI)
        self.assertEqual(nir_neuron.tau_mem[0], 0.02)
        self.assertEqual(nir_neuron.tau_syn[0], 0.005)
        self.assertEqual(nir_neuron.v_leak[0], 0.1)
        self.assertEqual(nir_neuron.tau_mem.shape, (10,))

    def test_cubalif_to_nir(self):
        exp = hxsnn.Experiment(mock=True)
        neuron = hxsnn.LIF(
            size=10,
            experiment=exp,
            leak=hxsnn.MixedHXModelParameter(0.1, 80),
            reset=hxsnn.MixedHXModelParameter(0.5, 80),
            threshold=hxsnn.MixedHXModelParameter(1.0, 150),
            tau_mem=0.02,
            tau_syn=0.005,
            trace_scale=1.0,
            cadc_time_shift=0.001,
            shift_cadc_to_first=True,
            enable_cadc_recording=True
        )

        nir_neuron = _map_hxtorch_to_nir(neuron)

        self.assertIsInstance(nir_neuron, nir.CubaLIF)
        self.assertEqual(nir_neuron.tau_mem[0], 0.02)
        self.assertEqual(nir_neuron.tau_syn[0], 0.005)
        self.assertEqual(nir_neuron.v_leak[0], 0.1)
        self.assertEqual(nir_neuron.v_threshold[0], 1.0)
        self.assertEqual(nir_neuron.tau_mem.shape, (10,))

    def test_synapse_to_nir(self):
        exp = hxsnn.Experiment(mock=True)
        weight = np.random.rand(5, 10).astype(np.float32)
        synapse = hxsnn.Synapse(
            in_features=10,
            out_features=5,
            experiment=exp
        )
        synapse.weight.data = torch.from_numpy(weight)

        nir_synapse = _map_hxtorch_to_nir(synapse)

        self.assertIsInstance(nir_synapse, nir.Linear)
        self.assertEqual(nir_synapse.weight.shape, (5, 10))
        self.assertTrue(np.array_equal(nir_synapse.weight, weight))

    def test_network_to_nir(self):
        mysnn = SNN(n_in=5,
                    n_hidden=120,
                    n_out=3,
                    mock=True)
        input_sample = torch.randn(5)

        nir_graph = to_nir(mysnn, input_sample)


class TestFromNIRConversion(unittest.TestCase):

    def test_cubali_from_nir(self):
        exp = hxsnn.Experiment(mock=True)
        cfg = ConversionConfig()
        nir_neuron = nir.CubaLI(
            tau_mem=np.array([0.02] * 10),
            tau_syn=np.array([0.005] * 10),
            r=np.array([1.0] * 10),
            v_leak=np.array([0.1] * 10)
        )

        hxtorch_neuron = _map_nir_to_hxtorch(exp, nir_neuron, cfg)

        self.assertIsInstance(hxtorch_neuron, hxsnn.LI)
        self.assertEqual(hxtorch_neuron.tau_mem.model_value, 0.02)
        self.assertEqual(hxtorch_neuron.tau_syn.model_value, 0.005)
        self.assertEqual(hxtorch_neuron.leak.model_value, 0.1)
        self.assertEqual(hxtorch_neuron.size, 10)

    def test_cubalif_from_nir(self):
        exp = hxsnn.Experiment(mock=True)
        cfg = ConversionConfig()
        nir_neuron = nir.CubaLIF(
            tau_mem=np.array([0.02] * 10),
            tau_syn=np.array([0.005] * 10),
            r=np.array([1.0] * 10),
            v_leak=np.array([0.1] * 10),
            v_reset=np.array([0.0] * 10),
            v_threshold=np.array([1.0] * 10)
        )

        hxtorch_neuron = _map_nir_to_hxtorch(exp, nir_neuron, cfg)

        self.assertIsInstance(hxtorch_neuron, hxsnn.LIF)
        self.assertEqual(hxtorch_neuron.tau_mem.model_value, 0.02)
        self.assertEqual(hxtorch_neuron.tau_syn.model_value, 0.005)
        self.assertEqual(hxtorch_neuron.leak.model_value, 0.1)
        self.assertEqual(hxtorch_neuron.reset.model_value, 0.0)
        self.assertEqual(hxtorch_neuron.threshold.model_value, 1.0)
        self.assertEqual(hxtorch_neuron.size, 10)

    def test_synapse_from_nir(self):
        exp = hxsnn.Experiment(mock=True)
        cfg = ConversionConfig()
        weight = np.random.rand(5, 10).astype(np.float32)
        nir_synapse = nir.Linear(weight)

        hxtorch_synapse = _map_nir_to_hxtorch(exp, nir_synapse, cfg)

        self.assertIsInstance(hxtorch_synapse, hxsnn.Synapse)
        self.assertEqual(hxtorch_synapse.weight.shape, (5, 10))
        self.assertTrue(np.array_equal(hxtorch_synapse.weight.detach().numpy(), weight))

    def test_network_from_nir(self):
        exp = hxsnn.Experiment(mock=True)
        cfg = ConversionConfig()

        nir_graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=np.array([5])),
                "linear": nir.Linear(weight=np.random.rand(10, 5)),
                "lif": nir.CubaLIF(
                    tau_mem=np.array([0.02] * 10),
                    tau_syn=np.array([0.005] * 10),
                    r=np.array([1.0] * 10),
                    v_leak=np.array([0.1] * 10),
                    v_reset=np.array([0.0] * 10),
                    v_threshold=np.array([1.0] * 10)
                ),
                "output": nir.Output(output_type=np.array([10]))
            },
            edges=[
                ("input", "linear"),
                ("linear", "lif"),
                ("lif", "output")
            ]
        )

        hxtorch_network = hxsnn.from_nir(nir_graph, cfg)

        sample_input = {"input": torch.randint(0, 2, (10, 100, 5),
                                               dtype=torch.float32)}
        output = hxtorch_network(sample_input)

if __name__ == '__main__':
    unittest.main()
