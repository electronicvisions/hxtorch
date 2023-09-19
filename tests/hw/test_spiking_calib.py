"""
Test SNN examples
"""
import unittest
from functools import partial
import torch
import numpy as np
import quantities as pq

from dlens_vx_v3 import halco
from calix.spiking.neuron import NeuronCalibTarget

from hxtorch.spiking.calibrated_params import CalibratedParams


class TestCalibratedParams(unittest.TestCase):
    """ Tests implicit neuron calibration """

    def test_init(self) -> None:
        """ Test construction without errors """
        # Default
        CalibratedParams()

        # Fill
        CalibratedParams(
            leak=torch.as_tensor(80),
            reset=torch.as_tensor(80),
            threshold=torch.as_tensor(125),
            tau_mem=torch.as_tensor(10.),
            tau_syn=torch.as_tensor(10.),
            i_synin_gm=torch.as_tensor(500),
            e_coba_reversal=torch.as_tensor(500),
            e_coba_reference=torch.as_tensor(500),
            membrane_capacitance=torch.as_tensor(63),
            refractory_time=torch.as_tensor(2.),
            synapse_dac_bias=torch.as_tensor(600),
            holdoff_time=torch.as_tensor(0))

    def test_from_calix_targets(self) -> None:
        # Some logical neurons
        neurons = [
            halco.LogicalNeuronOnDLS(
                halco.LogicalNeuronCompartments(
                    {halco.CompartmentOnLogicalNeuron():
                     [halco.AtomicNeuronOnLogicalNeuron()]}),
                     halco.AtomicNeuronOnDLS(halco.EnumRanged_512_(i)))
                     for i in range(23, 48)]
        coords = [
            an.toEnum().value() for neuron in neurons
            for an in neuron.get_atomic_neurons()]
        selector = [
            neuron.get_atomic_neurons()[0].toEnum().value()
            for neuron in neurons]

        # Default
        target = NeuronCalibTarget.DenseDefault
        params = CalibratedParams()
        params.from_calix_targets(target, neurons)
        self.check_params(target, params, selector, coords)

        # All numbers
        target = NeuronCalibTarget(
            leak=80,
            reset=70,
            threshold=125,
            tau_mem=10. * pq.us,
            tau_syn=10. * pq.us,
            i_synin_gm=500,
            e_coba_reversal=None,
            e_coba_reference=None,
            membrane_capacitance=63,
            refractory_time=2. * pq.us,
            synapse_dac_bias=600,
            holdoff_time=0 * pq.us)
        params = CalibratedParams()
        params.from_calix_targets(target, neurons)
        self.check_params(target, params, selector, coords)

        # Some larger logical neurons
        neurons = [
            halco.LogicalNeuronOnDLS(
                halco.LogicalNeuronCompartments(
                    {halco.CompartmentOnLogicalNeuron():
                     [halco.AtomicNeuronOnLogicalNeuron(
                         halco.EnumRanged_256_(0)),
                      halco.AtomicNeuronOnLogicalNeuron(
                          halco.EnumRanged_256_(1))]}),
                     halco.AtomicNeuronOnDLS(halco.EnumRanged_512_(2 * i)))
                     for i in range(25)]
        coords = [
            an.toEnum().value() for neuron in neurons
            for an in neuron.get_atomic_neurons()]
        selector = [
            neuron.get_atomic_neurons()[0].toEnum().value()
            for neuron in neurons]

        params = CalibratedParams()
        params.from_calix_targets(target, neurons)
        self.check_params(target, params, selector, coords)

        target = NeuronCalibTarget(
            leak=80,
            reset=70,
            threshold=125,
            tau_mem=10. * pq.us,
            tau_syn=10. * pq.us,
            i_synin_gm=500,
            e_coba_reversal=None,
            e_coba_reference=None,
            membrane_capacitance=63,
            refractory_time=2. * pq.us,
            synapse_dac_bias=600,
            holdoff_time=0 * pq.us)
        params = CalibratedParams()
        params.from_calix_targets(target, neurons)
        self.check_params(target, params, selector, coords)

    def test_to_calix_targets(self) -> None:
        # Default target as ExecutionInstance does
        # Some logical neurons
        neurons = [
            halco.LogicalNeuronOnDLS(
                halco.LogicalNeuronCompartments(
                    {halco.CompartmentOnLogicalNeuron():
                     [halco.AtomicNeuronOnLogicalNeuron()]}),
                     halco.AtomicNeuronOnDLS(halco.EnumRanged_512_(i)))
                     for i in range(23, 48)]
        coords = [
            an.toEnum().value() for neuron in neurons
            for an in neuron.get_atomic_neurons()]
        size = len(neurons)

        # All numbers, non-default
        params = CalibratedParams(
            leak=110,
            reset=120,
            threshold=130,
            tau_mem=2e-5,
            tau_syn=3e-5,
            i_synin_gm=600,
            e_coba_reversal=None,
            e_coba_reference=None,
            membrane_capacitance=55,
            refractory_time=3e-6,
            synapse_dac_bias=700,
            holdoff_time=1)
        target = NeuronCalibTarget.DenseDefault
        target.synapse_dac_bias = None
        target.i_synin_gm = np.array([None, None])
        params.to_calix_targets(target, neurons)
        self.check_targets(target, params, coords)

        # All tensors, non-default
        params = CalibratedParams(
            leak=torch.tensor(110),
            reset=torch.tensor(120),
            threshold=torch.tensor(130),
            tau_mem=torch.tensor(2e-5),
            tau_syn=torch.tensor(3e-5),
            i_synin_gm=torch.tensor(600),
            e_coba_reversal=None,
            e_coba_reference=None,
            membrane_capacitance=torch.tensor(55),
            refractory_time=torch.tensor(3e-6),
            synapse_dac_bias=torch.tensor(700),
            holdoff_time=torch.tensor(1))
        target = NeuronCalibTarget.DenseDefault
        target.synapse_dac_bias = None
        target.i_synin_gm = np.array([None, None])
        params.to_calix_targets(target, neurons)
        self.check_targets(target, params, coords)

        # All tensors of size, non-default
        params = CalibratedParams(
            leak=torch.full((size,), 110),
            reset=torch.full((size,), 120),
            threshold=torch.full((size,), 130),
            tau_mem=torch.full((size,), 2e-5),
            tau_syn=torch.full((2, size,), 3e-5),
            i_synin_gm=torch.full((2,), 600),
            e_coba_reversal=None,
            e_coba_reference=None,
            membrane_capacitance=torch.full((size,), 55),
            refractory_time=torch.full((size,), 3e-6),
            synapse_dac_bias=torch.tensor(700),
            holdoff_time=torch.full((size,), 1))
        target = NeuronCalibTarget.DenseDefault
        target.synapse_dac_bias = None
        target.i_synin_gm = np.array([None, None])
        params.to_calix_targets(target, neurons)
        self.check_targets(target, params, coords)

        target.i_synin_gm = np.array([400, None])
        self.assertRaises(
            AttributeError, partial(params.to_calix_targets, target, neurons))

        target.synapse_dac_bias = 600
        self.assertRaises(
            AttributeError, partial(params.to_calix_targets, target, neurons))

    def check_params(self, target, params, selector, coords):
        for key in ["tau_mem", "refractory_time", "holdoff_time",
                    "leak", "reset", "threshold", "membrane_capacitance"]:
            self.assertTrue(
                torch.equal(
                    torch.tensor([len(selector)]),
                    torch.tensor(getattr(params, key).shape)))
            if (not isinstance(getattr(target, key), np.ndarray)
                or not getattr(target, key).shape):
                self.assertTrue(
                    torch.all(
                        torch.tensor(getattr(target, key))
                        == getattr(params, key)))
            elif getattr(target, key).shape == (halco.AtomicNeuronOnDLS.size,):
                self.assertTrue(
                    torch.equal(
                        torch.tensor(
                            getattr(target, key)[selector]),
                            getattr(params, key)))

        # tau_syn
        if (not isinstance(target.tau_syn, np.ndarray)
            or not target.tau_syn.shape):
            self.assertTrue(
                torch.equal(
                    torch.tensor([len(selector)]),
                    torch.tensor(params.tau_syn.shape)))
            self.assertTrue(
                torch.all(
                    torch.tensor(target.tau_syn) == params.tau_syn))
        elif target.tau_syn.shape == (
            halco.SynapticInputOnNeuron.size, halco.AtomicNeuronOnDLS.size):
            self.assertTrue(
                torch.equal(
                    torch.tensor(
                        [halco.SynapticInputOnNeuron.size, len(selector)]),
                    torch.tensor(params.tau_syn.shape)))
            self.assertTrue(
                torch.equal(
                    torch.tensor(target.tau_syn[:, selector]), params.tau_syn))
        elif target.tau_syn.shape == (halco.SynapticInputOnNeuron.size,):
            self.assertTrue(
                torch.equal(
                    torch.tensor([len(selector)]),
                    torch.tensor(params.tau_syn.shape)))
            self.assertTrue(
                torch.equal(
                    torch.tensor(target.tau_syn), params.tau_syn))
        elif target.tau_syn.shape == (halco.AtomicNeuronOnDLS.size,):
            self.assertTrue(
                torch.equal(
                    torch.tensor([len(selector)]),
                    torch.tensor(params.tau_syn.shape)))
            self.assertTrue(
                torch.equal(
                    torch.tensor(target.tau_syn)[selector], params.tau_syn))

        for key in ["e_coba_reversal", "e_coba_reference"]:
            if getattr(target, key) is None:
                self.assertIsNone(getattr(params, key))
            else:
                self.assertTrue(
                    torch.equal(
                        torch.tensor(
                            [halco.SynapticInputOnNeuron.size, len(selector)]),
                        torch.tensor(getattr(params, key).shape)))
                if getattr(target, key).shape == (
                    halco.SynapticInputOnNeuron.size,):
                    if all(torch.isnan(getattr(target, key))):
                        self.assertTrue(all(torch.isnan(getattr(params, key))))
                    else:
                        self.assertTrue(
                            torch.all(torch.tensor(getattr(target, key)))
                            == getattr(params, key))
                else:
                    if torch.all(
                        torch.isnan(
                            torch.tensor(getattr(target, key))[:, coords])):
                        self.assertTrue(
                            torch.all(torch.isnan(getattr(params, key))))
                    else:
                        self.assertTrue(
                            torch.all(
                                torch.tensor(
                                    getattr(target, key))[:, selector]
                                    == getattr(params, key)))

    def check_targets(self, target, params, coords):
        # Leak
        for key in ["leak",
                    "reset",
                    "threshold",
                    "membrane_capacitance"]:
            this_target = getattr(target, key)
            this_param = getattr(params, key)
            self.assertTrue(
                torch.equal(
                    torch.tensor(this_target.shape),
                    torch.tensor([halco.AtomicNeuronOnDLS.size])))
            self.assertTrue(
                torch.all(torch.tensor(this_target[coords]) == this_param))

        for key in ["tau_mem",
                    "refractory_time",
                    "holdoff_time"]:
            this_target = getattr(target, key)
            this_param = getattr(params, key)
            self.assertTrue(
                torch.equal(
                    torch.tensor(this_target.shape),
                    torch.tensor([halco.AtomicNeuronOnDLS.size])))
            self.assertTrue(
                torch.all(
                    torch.tensor(this_target[coords]) == this_param * 1e6))

        # e_coba_reversal
        self.assertTrue(
            torch.equal(
                torch.tensor(target.e_coba_reversal.shape),
                torch.tensor([2, halco.AtomicNeuronOnDLS.size])))
        if params.e_coba_reversal is not None:
            self.assertTrue(
                torch.all(
                    torch.tensor(target.e_coba_reversal[:, coords])
                    == params.e_coba_reversal))
        else:
            self.assertTrue(
                torch.all(
                    torch.tensor(target.e_coba_reversal[:, coords])
                    == torch.tensor([[torch.inf], [-torch.inf]])))

        # e_coba_reference
        self.assertTrue(
            torch.equal(
                torch.tensor(target.e_coba_reference.shape),
                torch.tensor(
                    [halco.SynapticInputOnNeuron.size,
                     halco.AtomicNeuronOnDLS.size])))
        if params.e_coba_reference is not None:
            self.assertTrue(
                torch.all(
                    torch.tensor(target.e_coba_reference[:, coords])
                    == params.e_coba_reference))
        else:
            self.assertTrue(
                torch.all(torch.isnan(torch.tensor(
                    target.e_coba_reference[:, coords]))))

        # tau syn
        self.assertTrue(
            torch.equal(
                torch.tensor(target.tau_syn.shape),
                torch.tensor([halco.SynapticInputOnNeuron.size,
                              halco.AtomicNeuronOnDLS.size])))
        self.assertTrue(
            torch.all(
                torch.tensor(
                    target.tau_syn[:, coords]) == params.tau_syn * 1e6))

        # i_synin_gm
        self.assertTrue(
            torch.equal(
                torch.tensor(
                    target.i_synin_gm.shape),
                    torch.tensor([halco.SynapticInputOnNeuron.size])))
        self.assertTrue(
            torch.all(
                torch.tensor(target.i_synin_gm) == params.i_synin_gm))

        # synapse_dac_bias
        self.assertEqual(target.synapse_dac_bias, params.synapse_dac_bias)


if __name__ == "__main__":
    unittest.main()
