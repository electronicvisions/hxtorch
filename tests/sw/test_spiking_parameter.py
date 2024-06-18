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

from hxtorch.spiking.experiment import Experiment
from hxtorch.spiking.parameter import (
    HXParameter, MixedHXModelParameter, HXTransformedModelParameter,
    ModelParameter)


class TestParameter(unittest.TestCase):
    """ Tests implicit neuron calibration """

    def test_hxparameter(self) -> None:
        """ Test HXParameter """
        param = HXParameter(80.)
        self.assertEqual(param.hardware_value, 80.)
        self.assertEqual(param.model_value, 80.)

        param.hardware_value = 10.
        self.assertEqual(param.hardware_value, 10.)
        self.assertEqual(param.model_value, 10.)

    def test_mixedhxmodelparameter(self) -> None:
        """ Test MixedHXModelParameter """
        param = MixedHXModelParameter(1., 80.)
        self.assertEqual(param.hardware_value, 80.)
        self.assertEqual(param.model_value, 1.)

        param.hardware_value = 10.
        param.model_value = 80.
        self.assertEqual(param.hardware_value, 10.)
        self.assertEqual(param.model_value, 80.)

    def test_hxtransformedmodelparameter(self) -> None:
        """ Test HXTransformedModelParameter """
        param = HXTransformedModelParameter(1., lambda x: 80. * x)
        self.assertEqual(param.hardware_value, 80.)
        self.assertEqual(param.model_value, 1.)

        param.model_value = 2.
        self.assertEqual(param.hardware_value, 160.)
        self.assertEqual(param.model_value, 2.)

    def test_modelparameter(self) -> None:
        """ Test ModelParameter """
        param = ModelParameter(1.)
        self.assertEqual(param.hardware_value, 1.)
        self.assertEqual(param.model_value, 1.)

        param.model_value = 2.
        self.assertEqual(param.hardware_value, 2.)
        self.assertEqual(param.model_value, 2.)


if __name__ == "__main__":
    unittest.main()
