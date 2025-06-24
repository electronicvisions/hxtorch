"""
Measure translation between hardware and software model
"""
from typing import Optional, Tuple, Dict
import pylogging as logger

import torch

from dlens_vx_v3 import halco

import hxtorch
import hxtorch.spiking as hxsnn
from hxtorch.spiking.morphology import Morphology, SingleCompartmentNeuron
from hxtorch.spiking.utils.dynamic_range.helper import ConstantCurrentNeuron
from hxtorch.spiking.parameter import HXBaseParameter


class Threshold:
    """ Class to measure thresholds on BSS-2 """

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
        self.neuron = None
        self.traces = None
        self.thresholds = None
        self.thresholds_mean = None

        # TODO: This probably fails if morphology is more complex
        self.output_size = halco.AtomicNeuronOnDLS.size // sum(
            [len(ans) for ans in
             neuron_structure.compartments.get_compartments().values()])

        self.log = logger.get("hxtorch.spiking.utils.Thresholds")

    def build_model(self, inputs: torch.Tensor, exp: hxsnn.Experiment,
                    enable_current: bool = True) \
            -> hxsnn.TensorHandle:
        """ Build model to map to hardware """
        self.log.TRACE("Build model ...")

        # Layers
        synapse = hxsnn.Synapse(
            1, self.output_size, experiment=exp)
        self.neuron = ConstantCurrentNeuron(
            self.output_size, **self.params, experiment=exp,
            shift_cadc_to_first=False, neuron_structure=self.neuron_structure)
        self.neuron.enable_current = enable_current

        # forward
        inputs = hxsnn.LIFObservables(spikes=inputs)
        currents = synapse(inputs)
        traces = self.neuron(currents)
        return traces

    # pylint: disable=invalid-name
    def run(self, dt: float = 1e-6):
        """ Execute forward """
        self.log.TRACE("Run experiment ...")
        hxtorch.init_hardware()

        # Run for baseline
        exp = hxsnn.Experiment(mock=False, dt=dt)
        # Load calib
        if self.calib_path is not None:
            exp.default_execution_instance.load_calib(self.calib_path)

        inputs = torch.zeros((self.time_length, self.batch_size, 1))
        baselines = self.build_model(inputs, exp, enable_current=False)
        hxsnn.run(exp, self.time_length)

        # Run for threshold
        exp = hxsnn.Experiment(mock=False, dt=dt)
        # Load calib
        if self.calib_path is not None:
            exp.default_execution_instance.load_calib(self.calib_path)

        # Explicitly load because we want to set initial config
        inputs = torch.zeros((self.time_length, self.batch_size, 1))
        traces = self.build_model(inputs, exp)

        # run
        hxsnn.run(exp, self.time_length)
        hxtorch.release_hardware()
        self.log.TRACE("Experiment ran.")

        return self.post_process(traces, baselines)

    def post_process(self, traces: torch.Tensor, baselines: torch.Tensor) \
            -> Tuple[torch.Tensor, ...]:
        """ post-process data """
        self.log.TRACE("Postprocessing ...")

        # Make sure we have spikes
        non_spiking_entries = (traces.spikes.sum(0) < 1).nonzero()
        if non_spiking_entries.shape[0] > 0:
            non_spiking_neurons = torch.unique(
                non_spiking_entries[:, 1])[0].tolist()
            self.log.WARN(
                "No spikes measured in all or some batch entries for "
                + f"neuron(s) {non_spiking_neurons}.")

        baselines = baselines.membrane_cadc.detach().mean(0).mean(0)
        self.traces = traces.membrane_cadc.detach() - baselines
        self.thresholds = torch.max(self.traces, 0)[0].mean(0)
        self.thresholds_mean = self.thresholds.mean()
        return self.thresholds_mean, self.thresholds, self.traces, \
            self.hw_neurons


def get_trace_scaling(
        params: Dict[str, HXBaseParameter],
        calib_path: Optional[str] = None,
        neuron_structure: Morphology = SingleCompartmentNeuron(1)):
    threshold_runner = Threshold(
        params, calib_path, neuron_structure=neuron_structure)
    threshold_hw, _, _, _ = threshold_runner.run()
    return threshold_runner.neuron.threshold.model_value / threshold_hw


if __name__ == "__main__":
    trace_scale = get_trace_scaling()
