"""
Measure translation between hardware and software weights
"""
from typing import Tuple, Optional, Union, Dict
import pylogging as logger
from tqdm import tqdm

import torch
from scipy.optimize import curve_fit

from dlens_vx_v3 import lola, halco

import hxtorch
import hxtorch.spiking as hxsnn
import hxtorch.spiking.functional as F
from hxtorch.spiking.morphology import Morphology, SingleCompartmentNeuron
from hxtorch.spiking.utils.dynamic_range.threshold import Threshold
from hxtorch.spiking.parameter import HXBaseParameter


class WeightScaling:
    """
    Class to measure weight scaling between SW weight and the weight on BSS-2
    """

    max_weight = lola.SynapseWeightMatrix.Value.max
    min_weight = -lola.SynapseWeightMatrix.Value.max

    # pylint: disable=too-many-arguments
    def __init__(self, params: Dict[str, HXBaseParameter] = {},
                 calib_path: str = None,
                 batch_size: int = 100, trace_scale: float = 1.,
                 neuron_structure: Morphology = SingleCompartmentNeuron(1)):
        """ """
        self.calib_path = calib_path
        self.batch_size = batch_size
        self.neuron_structure = neuron_structure
        self.trace_scale = trace_scale
        self.params = params
        # Constants
        self.time_length = 100
        self.hw_neurons = None

        # TODO: This probably fails if morphology is complexer
        self.output_size = halco.AtomicNeuronOnDLS.size // sum(
            [len(ans) for ans in
             neuron_structure.compartments.get_compartments().values()])

        self.log = logger.get("hxtorch.spiking.utils.WeightScaling")

    # pylint: disable=arguments-differ, attribute-defined-outside-init
    def execute(self, weight: int, mock: bool = False) -> torch.Tensor:
        """ Execute forward """
        self.log.TRACE(f"Run experiment with weight {weight} ...")

        # Instance
        if mock:
            inputs = torch.zeros(self.time_length, 1, 1)
        else:
            inputs = torch.zeros(self.time_length, self.batch_size, 1)
        inputs[10, :, :] = 1
        self.synapse.weight.data.fill_(weight)

        # forward
        spikes = hxsnn.NeuronHandle(spikes=inputs)
        currents = self.synapse(spikes)
        traces = self.neuron(currents)

        hxsnn.run(self.exp, self.time_length)
        self.log.TRACE("Experiment ran.")

        return self.post_process(traces, weight)

    def post_process(self, traces: torch.Tensor, weight: float) \
            -> Tuple[torch.Tensor, ...]:
        """ post-process data """
        self.log.TRACE("Postprocessing ...")
        self.traces = traces.membrane_cadc.detach() - self.baselines
        # Get max/min PSP over time
        if weight >= 0:
            self.amp = self.traces.max(0)[0].mean(0)
        else:
            self.amp = self.traces.min(0)[0].mean(0)
        self.amp_mean = self.amp.mean()
        return self.amp_mean, self.amp, self.traces

    # pylint: disable=arguments-differ, too-many-arguments, too-many-locals
    def run(self, weight_step: int = 1) -> Tuple[torch.Tensor, ...]:
        """ run all measurements and compute output """
        hxtorch.init_hardware()

        # Sweep weights
        self.log.INFO(f"Using weight step: {weight_step}")
        hw_weights = torch.linspace(
            self.min_weight, self.max_weight, weight_step, dtype=int)

        # HW layers
        self.exp = hxsnn.Experiment(mock=False, dt=1e-6)
        self.synapse = hxsnn.Synapse(
            1, self.output_size, experiment=self.exp)
        self.neuron = hxsnn.ReadoutNeuron(
            self.output_size, **self.params, experiment=self.exp,
            neuron_structure=self.neuron_structure,
            trace_scale=self.trace_scale,
            shift_cadc_to_first=True)

        # Load calib
        if self.calib_path is not None:
            self.exp.default_execution_instance.load_calib(self.calib_path)

        # Sweep
        hw_amps = torch.zeros(hw_weights.shape[0], self.output_size)
        self.baselines = 0
        self.log.INFO("Begin hardware weight sweep...")
        pbar = tqdm(total=hw_weights.shape[0])
        for i, weight in enumerate(hw_weights):
            # Measure
            _, hw_amps[i], _ = self.execute(weight)
            # Update
            pbar.set_postfix(
                weight=f"{weight}", mean_amp=float(hw_amps[i].mean()))
            pbar.update()
        pbar.close()
        self.log.INFO("Hardware weight sweep finished.")

        # Fit
        self.log.INFO("Fit hardware data...")
        hw_scales = torch.zeros(self.output_size)
        for nrn in range(self.output_size):
            popt, _ = curve_fit(
                f=lambda x, a: a * x, xdata=hw_weights.numpy(),
                ydata=hw_amps[:, nrn].numpy())
            hw_scales[nrn] = popt[0]

        # Mock values
        self.exp = hxsnn.Experiment(mock=True, dt=1e-6)
        self.synapse = hxsnn.Synapse(
            1, self.output_size, experiment=self.exp)
        self.neuron = hxsnn.ReadoutNeuron(
            self.output_size, **self.params, experiment=self.exp,
            trace_scale=self.trace_scale,
            shift_cadc_to_first=False)

        self.log.INFO("Begin mock weight sweep...")
        sw_weights = torch.arange(-1, 1 + 0.1, 0.1)
        # Software amplitudes
        self.baselines = self.neuron.leak.model_value
        sw_amps = torch.zeros(sw_weights.shape[0], self.output_size)
        pbar = tqdm(total=sw_weights.shape[0])
        for i, weight in enumerate(sw_weights):
            # Measure
            _, sw_amps[i], _ = self.execute(weight, mock=True)
            pbar.set_postfix(
                weight=f"{weight}", mean_amp=float(sw_amps[i].mean()))
            pbar.update()
        pbar.close()

        # SW scale
        sw_scales = torch.zeros(self.output_size)
        for nrn in range(self.output_size):
            popt, _ = curve_fit(
                f=lambda x, a: a * x, xdata=sw_weights.numpy(),
                ydata=sw_amps[:, nrn].numpy())
            sw_scales[nrn] = popt[0]

        # Resulting scales
        scales = sw_scales / hw_scales

        self.log.INFO(
            f"Mock scale: {sw_scales}, HW scale: {hw_scales.mean()}"
            + f" +- {hw_scales.std()}")
        self.log.INFO(f"SW -> HW translation factor: {scales.mean()}")

        hxtorch.release_hardware()

        return (
            scales.mean(), scales, hw_scales, sw_scales, hw_amps, sw_amps)


def get_weight_scaling(
        params: Dict[str, HXBaseParameter] = {},
        weight_step: int = 1, calib_path: Optional[str] = None,
        neuron_structure: SingleCompartmentNeuron
        = SingleCompartmentNeuron(1)) -> float:
    scale_runner = WeightScaling(
        params, calib_path, neuron_structure=neuron_structure)
    weight_scale, _, _, _, _, _ = scale_runner.run(weight_step=weight_step)
    threshold_runner = Threshold(
        params, calib_path, neuron_structure=neuron_structure)
    threshold_hw, _, _, _ = threshold_runner.run()

    # Compute effective weight scaling between SW and HW weights
    return weight_scale * (
        threshold_hw / threshold_runner.neuron.threshold.model_value)


if __name__ == "__main__":
    weight_scale = get_weight_scaling(weight_step=10)
