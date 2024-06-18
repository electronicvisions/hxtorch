"""
Test SNN examples
"""
import unittest
from functools import partial
import torch
import numpy as np
import quantities as pq

from dlens_vx_v3 import halco
from calix.spiking import SpikingCalibTarget
from calix.spiking.neuron import NeuronCalibTarget

from hxtorch.spiking.experiment import Experiment
from hxtorch.spiking.modules.types import Population
from hxtorch.spiking.parameter import HXParameter


class TestPopulation(unittest.TestCase):

    def test_params_from_calibration(self) -> None:
        """ Test conversion from calix targets to params """
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
        target = SpikingCalibTarget()
        target.neuron = NeuronCalibTarget.DenseDefault

        nrn = Population(25, experiment=Experiment())
        nrn.params_from_calibration(target, neurons)
        self.check_params(
            target.neuron_target, nrn.params_dict(), selector, coords)

        # All numbers
        nrn = Population(25, experiment=Experiment())
        nrn.leak=HXParameter(80)
        nrn.reset=HXParameter(70)
        nrn.threshold=HXParameter(125)
        nrn.tau_mem=HXParameter(10e-6)
        nrn.tau_syn=HXParameter(10e-6)
        nrn.i_synin_gm=HXParameter(500)
        nrn.e_coba_reversal=HXParameter(None)
        nrn.e_coba_reference=HXParameter(None)
        nrn.membrane_capacitance=HXParameter(63)
        nrn.refractory_time=HXParameter(2e-6)
        nrn.synapse_dac_bias=HXParameter(600)
        nrn.holdoff_time=HXParameter(0e-6)

        nrn.params_from_calibration(target, neurons)
        self.check_params(
            target.neuron_target, nrn.params_dict(), selector, coords)

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

        nrn = Population(25, experiment=Experiment())
        nrn.params_from_calibration(target, neurons)
        self.check_params(
            target.neuron_target, nrn.params_dict(), selector, coords)

        nrn = Population(25, experiment=Experiment())
        nrn.leak=HXParameter(80)
        nrn.reset=HXParameter(70)
        nrn.threshold=HXParameter(125)
        nrn.tau_mem=HXParameter(10e-6)
        nrn.tau_syn=HXParameter(10e-6)
        nrn.i_synin_gm=HXParameter(500)
        nrn.e_coba_reversal=HXParameter(None)
        nrn.e_coba_reference=HXParameter(None)
        nrn.membrane_capacitance=HXParameter(63)
        nrn.refractory_time=HXParameter(2e-6)
        nrn.synapse_dac_bias=HXParameter(600)
        nrn.holdoff_time=HXParameter(0e-6)

        nrn.params_from_calibration(target, neurons)
        self.check_params(
            target.neuron_target, nrn.params_dict(), selector, coords)

    def test_calibration_from_params(self) -> None:
        """ Test conversion from params to calix targets """
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
        nrn = Population(25, experiment=Experiment())
        nrn.leak=HXParameter(110)
        nrn.reset=HXParameter(120)
        nrn.threshold=HXParameter(130)
        nrn.tau_mem=HXParameter(2e-6)
        nrn.tau_syn=HXParameter(3e-6)
        nrn.i_synin_gm=HXParameter(600)
        nrn.e_coba_reversal=HXParameter(None)
        nrn.e_coba_reference=HXParameter(None)
        nrn.membrane_capacitance=HXParameter(55)
        nrn.refractory_time=HXParameter(3e-6)
        nrn.synapse_dac_bias=HXParameter(700)
        nrn.holdoff_time=HXParameter(1e-6)
    
        target = SpikingCalibTarget()
        target.neuron_target = NeuronCalibTarget.DenseDefault
        target.neuron_target.synapse_dac_bias = None
        target.neuron_target.i_synin_gm = np.array([None, None])
        nrn.calibration_from_params(target, neurons)
        self.check_targets(
            target.neuron_target, nrn.params_dict(), coords)

        # All tensors, non-default
        nrn = Population(25, experiment=Experiment())
        nrn.leak=HXParameter(torch.tensor(110))
        nrn.reset=HXParameter(torch.tensor(120))
        nrn.threshold=HXParameter(torch.tensor(130))
        nrn.tau_mem=HXParameter(torch.tensor(2e-5))
        nrn.tau_syn=HXParameter(torch.tensor(3e-5))
        nrn.i_synin_gm=HXParameter(torch.tensor(600))
        nrn.e_coba_reversal=HXParameter(None)
        nrn.e_coba_reference=HXParameter(None)
        nrn.membrane_capacitance=HXParameter(torch.tensor(55))
        nrn.refractory_time=HXParameter(torch.tensor(3e-6))
        nrn.synapse_dac_bias=HXParameter(torch.tensor(700))
        nrn.holdoff_time=HXParameter(torch.tensor(1))

        target = SpikingCalibTarget()
        target.neuron_target = NeuronCalibTarget.DenseDefault
        target.neuron_target.synapse_dac_bias = None
        target.neuron_target.i_synin_gm = np.array([None, None])
        nrn.calibration_from_params(target, neurons)
        self.check_targets(target.neuron_target, nrn.params_dict(), coords)

        # All tensors of size, non-default
        nrn = Population(25, experiment=Experiment())
        nrn.leak=HXParameter(torch.full((size,), 110))
        nrn.reset=HXParameter(torch.full((size,), 120))
        nrn.threshold=HXParameter(torch.full((size,), 130))
        nrn.tau_mem=HXParameter(torch.full((size,), 2e-5))
        nrn.tau_syn=HXParameter(torch.full((2, size,), 3e-5))
        nrn.i_synin_gm=HXParameter(torch.full((2,), 600))
        nrn.e_coba_reversal=HXParameter(None)
        nrn.e_coba_reference=HXParameter(None)
        nrn.membrane_capacitance=HXParameter(torch.full((size,), 55))
        nrn.refractory_time=HXParameter(torch.full((size,), 3e-6))
        nrn.synapse_dac_bias=HXParameter(torch.tensor(700))
        nrn.holdoff_time=HXParameter(torch.full((size,), 1))

        target = SpikingCalibTarget()
        target.neuron_target = NeuronCalibTarget.DenseDefault
        target.neuron_target.synapse_dac_bias = None
        target.neuron_target.i_synin_gm = np.array([None, None])
        nrn.calibration_from_params(target, neurons)
        self.check_targets(target.neuron_target, nrn.params_dict(), coords)

        target.neuron_target.i_synin_gm = np.array([400, None])
        self.assertRaises(
            AttributeError,
            partial(nrn.calibration_from_params, target, neurons))

        target.neuron_target.synapse_dac_bias = 600
        self.assertRaises(
            AttributeError,
            partial(nrn.calibration_from_params, target, neurons))

    def check_params(self, target, params, selector, coords):
        """ checks if params have expected shape and values """
        for key in ["tau_mem", "refractory_time", "holdoff_time",
                    "leak", "reset", "threshold", "membrane_capacitance"]:
            self.assertTrue(
                torch.equal(
                    torch.tensor([len(selector)]),
                    torch.tensor(params[key].hardware_value.shape)))
            target_value = getattr(target, key)
            if isinstance(target_value, pq.Quantity):
                target_value = target_value.rescale(pq.s)
            if (not isinstance(getattr(target, key), np.ndarray)
                or not getattr(target, key).shape):
                self.assertTrue(
                    torch.all(
                        torch.tensor(target_value)
                        == params[key].hardware_value))
            elif target_value.shape == (halco.AtomicNeuronOnDLS.size,):
                self.assertTrue(
                    torch.equal(
                        torch.tensor(
                            target_value[selector]),
                            params[key].hardware_value))

        # tau_syn 
        if (not isinstance(target.tau_syn, np.ndarray)
            or not target.tau_syn.shape):
            self.assertTrue(
                torch.equal(
                    torch.tensor([len(selector)]),
                    torch.tensor(params["tau_syn"].hardware_value.shape)))
            self.assertTrue(
                torch.all(torch.tensor(target.tau_syn.rescale(pq.s))
                          == params["tau_syn"].hardware_value))
        elif target.tau_syn.shape == (
            halco.SynapticInputOnNeuron.size, halco.AtomicNeuronOnDLS.size):
            self.assertTrue(
                torch.equal(
                    torch.tensor(
                        [halco.SynapticInputOnNeuron.size, len(selector)]),
                    torch.tensor(params["tau_syn"].hardware_value.shape)))
            self.assertTrue(
                torch.equal(
                    torch.tensor(target.tau_syn[:, selector].rescale(pq.s)),
                    params["tau_syn"].hardware_value))
        elif target.tau_syn.shape == (halco.SynapticInputOnNeuron.size,):
            self.assertTrue(
                torch.equal(
                    torch.tensor([len(selector)]),
                    torch.tensor(params["tau_syn"].hardware_value.shape)))
            self.assertTrue(
                torch.equal(
                    torch.tensor(target.tau_syn.rescale(pq.s)),
                    params["tau_syn"].hardware_value))
        elif target.tau_syn.shape == (halco.AtomicNeuronOnDLS.size,):
            self.assertTrue(
                torch.equal(
                    torch.tensor([len(selector)]),
                    torch.tensor(params["tau_syn"].hardware_value.shape)))
            self.assertTrue(
                torch.equal(
                    torch.tensor(target.tau_syn)[selector],
                    params["tau_syn"].hardware_value))

        for key in ["e_coba_reversal", "e_coba_reference"]:
            if getattr(target, key) is None:
                self.assertIsNone(params[key])
            else:
                self.assertTrue(
                    torch.equal(
                        torch.tensor(
                            [halco.SynapticInputOnNeuron.size, len(selector)]),
                        torch.tensor(params[key].shape)))
                if getattr(target, key).shape == (
                    halco.SynapticInputOnNeuron.size,):
                    if all(torch.isnan(getattr(target, key))):
                        self.assertTrue(all(torch.isnan(params[key])))
                    else:
                        self.assertTrue(
                            torch.all(torch.tensor(getattr(target, key)))
                            == params[key])
                else:
                    if torch.all(
                        torch.isnan(
                            torch.tensor(getattr(target, key))[:, coords])):
                        self.assertTrue(
                            torch.all(torch.isnan(params[key])))
                    else:
                        self.assertTrue(
                            torch.all(
                                torch.tensor(
                                    getattr(target, key))[:, selector]
                                    == params[key]))

    def check_targets(self, target, params, coords):
        """ Checks if targets have expected shapes and values """
        # Leak
        for key in ["leak",
                    "reset",
                    "threshold",
                    "membrane_capacitance"]:
            this_target = getattr(target, key)
            this_param = params[key].hardware_value
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
            this_param = params[key].hardware_value
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
        if params["e_coba_reversal"].hardware_value is not None:
            self.assertTrue(
                torch.all(
                    torch.tensor(target.e_coba_reversal[:, coords])
                    == params["e_coba_reversal"].hardware_value))
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
        if params["e_coba_reference"].hardware_value is not None:
            self.assertTrue(
                torch.all(
                    torch.tensor(target.e_coba_reference[:, coords])
                    == params["e_coba_reference"].hardware_value))
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
                torch.tensor(target.tau_syn[:, coords])
                == params["tau_syn"].hardware_value * 1e6))

        # i_synin_gm
        self.assertTrue(
            torch.equal(
                torch.tensor(
                    target.i_synin_gm.shape),
                    torch.tensor([halco.SynapticInputOnNeuron.size])))
        self.assertTrue(
            torch.all(
                torch.tensor(target.i_synin_gm)
                == params["i_synin_gm"].hardware_value))

        # synapse_dac_bias
        self.assertEqual(
            target.synapse_dac_bias, params["synapse_dac_bias"].hardware_value)


if __name__ == "__main__":
    unittest.main()
