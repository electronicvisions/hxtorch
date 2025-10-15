""" Measure the upper and lower boundaries on hardware """
from typing import Optional, Tuple, Dict
import pylogging as logger

import torch

from dlens_vx_v3 import lola, halco
import hxtorch
import hxtorch.spiking as hxsnn
from hxtorch.core.utils import calib_helper
from hxtorch.core.parameter import HXBaseParameter
from hxtorch.core.morphology import (
    Morphology,
    SingleCompartmentNeuron,
)


class Boundaries:
    """
    Class to measure upper and lower membrane voltage boundaries on BSS-2
    """

    def __init__(self, params: Dict[str, HXBaseParameter] = {},
                 calib_path: Optional[str] = None,
                 batch_size: int = 100,
                 neuron_structure: Morphology = SingleCompartmentNeuron(1)):
        """ """
        self.calib_path = calib_path
        self.batch_size = batch_size
        self.neuron_structure = neuron_structure
        self.params = params
        # Constants
        self.time_length = 100
        self.hw_neurons = None
        self.input_size = 20

        # TODO: This probably fails if morphology is complexer
        self.output_size = halco.AtomicNeuronOnDLS.size // sum(
            [len(ans) for ans in
             neuron_structure.compartments.get_compartments().values()])

        self.log = logger.get("hxtorch.spiking.utils.Boundaries")

    def build_model(self, inputs: torch.Tensor, exp: hxsnn.Experiment,
                    current_type: lola.AtomicNeuron.ConstantCurrent.Type = None,
                    enable_current: bool = True) \
            -> hxsnn.TensorHandle:
        """ Build model to map to hardware """
        self.log.TRACE("Build hardware model")

        # Layers
        synapse = hxsnn.Synapse(
            self.input_size, self.output_size, experiment=exp)
        self.neuron = hxsnn.LI(
            self.output_size,
            experiment=exp,
            **self.params,
            neuron_structure=self.neuron_structure,
            shift_cadc_to_first=False,
            enable_constant_current=enable_current,
            current_type=current_type,
        )

        # forward
        inputs = hxsnn.LIFObservables(spikes=inputs)
        currents = synapse(inputs)
        traces = self.neuron(currents)
        return traces

    def run(self, dt: float = 1e-6):
        """ Execute forward """
        self.log.TRACE("Run experiment ...")
        hxtorch.init_hardware()

        # Baseline
        exp = hxsnn.Experiment(mock=False, dt=dt)
        if self.calib_path is not None:
            exp.calibration = calib_helper.fixture_calibration_from_file(
                self.calib_path,
            )
        inputs = torch.zeros(
            (self.time_length, self.batch_size, self.input_size))
        baselines = self.build_model(inputs, exp, enable_current=False)
        hxsnn.run(exp, self.time_length)

        # Upper
        exp = hxsnn.Experiment(mock=False, dt=dt)
        if self.calib_path is not None:
            exp.calibration = calib_helper.fixture_calibration_from_file(
                self.calib_path,
            )
        inputs = torch.zeros(
            (self.time_length, self.batch_size, self.input_size))
        upperlines = self.build_model(
            inputs, exp,
            current_type=lola.AtomicNeuron.ConstantCurrent.Type.source)
        hxsnn.run(exp, self.time_length)

        # Lower
        exp = hxsnn.Experiment(mock=False, dt=dt)
        if self.calib_path is not None:
            exp.calibration = calib_helper.fixture_calibration_from_file(
                self.calib_path,
            )
        inputs = torch.zeros(
            (self.time_length, self.batch_size, self.input_size))
        lowerlines = self.build_model(
            inputs, exp,
            current_type=lola.AtomicNeuron.ConstantCurrent.Type.sink)
        # Load calib
        hxsnn.run(exp, self.time_length)

        hxtorch.release_hardware()
        self.log.TRACE("Experiment ran ...")

        return self.post_process(baselines, upperlines, lowerlines)

    def post_process(self, baselines: torch.Tensor, upperlines: torch.Tensor,
                     lowerlines: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """ post-process data """
        self.log.TRACE("Postprocessing ...")
        baselines_ = baselines.membrane_cadc.detach().mean(0).mean(0).mean()
        upperlines_ = upperlines.membrane_cadc.detach()[
            int(0.8 * self.time_length):].mean(0).mean(0).mean()
        lowerlines_ = lowerlines.membrane_cadc.detach()[
            int(0.8 * self.time_length):].mean(0).mean(0).mean()

        return baselines_, upperlines_, lowerlines_


def get_dynamic_range(
        neuron_structure: Morphology = SingleCompartmentNeuron(1),
        calib_path: Optional[str] = None,
        params: Dict[str, HXBaseParameter] = {}) -> float:
    boundary_runner = Boundaries(
        params, calib_path, neuron_structure=neuron_structure)
    return boundary_runner.run()


if __name__ == "__main__":
    get_dynamic_range()
