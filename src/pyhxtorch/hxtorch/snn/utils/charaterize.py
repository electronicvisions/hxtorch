"""
Provide functionality to measure raw hardware characteristics:
    - leak potentials [CADC values]
    - upper CADC boundaries [CADC values]
    - lower CADC boundaries [CADC values]
    - thresholds [CADC values]
    - scaling factor between software and hardware weights
    - time shift between hardware spikes and CADC readouts
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union
from tqdm import tqdm

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import torch

import _hxtorch
import hxtorch
from hxtorch import snn
import hxtorch.snn.functional as F

from dlens_vx_v3 import lola, halco, hal


class SpikingNeuron(snn.Neuron):
    """ Spiking neuron adjustment to get plain CADC values """

    def __init__(self, shift_to_first: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shift_to_first = shift_to_first

    # pylint: disable=unused-argument
    def post_process(self, hw_spikes, hw_cadc, hw_madc):
        cadc = hw_cadc.to_raw()[0]
        if self.shift_to_first:
            cadc -= cadc[:, 0, :].clone().unsqueeze(1)
        else:
            cadc -= self.offset

        cadc *= self.scale

        return hw_spikes, cadc, None


class ReadoutNeuron(snn.ReadoutNeuron):
    """ Readout neuron adjustment to get plain CADC values """

    def __init__(self, shift_to_first: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shift_to_first = shift_to_first

    # pylint: disable=unused-argument
    def post_process(self, hw_spikes, hw_cadc, hw_madc):
        cadc = hw_cadc.to_raw()[0]
        if self.shift_to_first:
            cadc -= cadc[:, 0, :].clone().unsqueeze(1)
        else:
            cadc = cadc - self.offset

        cadc *= self.scale

        return cadc, None


class BaseExperiment(ABC):

    def __init__(self, calibration_path: Optional[str] = None,
                 batch_size: int = 100,
                 output_size: int = halco.AtomicNeuronOnDLS.size):
        self.calibration_path = calibration_path
        self.batch_size = batch_size
        self.time_length = 100
        self.output_size = output_size
        self.input_size = 1
        self.offset = torch.zeros(self.output_size)
        self.scale = torch.ones(self.output_size)
        self.logger = hxtorch.logger.get("hxtorch.utils.BaseExperiment")
        self.hw_neurons = None

    @abstractmethod
    def post_process(self, traces: torch.Tensor) -> np.ndarray:
        pass

    @abstractmethod
    def build_model(self, inputs: torch.Tensor, instance: snn.Instance,
                    transforms: Callable) -> snn.TensorHandle:
        pass

    def execute(self, inputs: torch.Tensor, transforms: Callable,
                enable_constant_current: bool = False, mock: bool = False,
                **kwargs) -> snn.TensorHandle:
        """ Execute forward """
        # Instance
        instance = snn.Instance(mock=mock, dt=kwargs.get("dt", 1e-6))
        # model output traces
        traces = self.build_model(
            inputs, instance, transforms, **kwargs)

        # Set initial config
        if enable_constant_current:
            config = _hxtorch.get_chip()
            for nrn in halco.iter_all(halco.AtomicNeuronOnDLS):
                config.neuron_block.atomic_neurons[nrn] \
                    .constant_current.i_offset = 1000
                config.neuron_block.atomic_neurons[nrn] \
                    .constant_current.enable = True
            instance.initial_config = config

        # run
        snn.run(instance, self.time_length)

        # Remember HW <-> SW mapping
        if not mock:
            self.hw_neurons = instance.neuron_placement.id2atomicneuron(
                self.neurons.unit_ids)  # pylint: disable=no-member

        # Post-process measurements
        results = self.post_process(traces)

        return results

    def run(self, *args, **kwargs) -> snn.TensorHandle:
        # Initialize hardware
        if not kwargs.get("mock", False):
            if self.calibration_path is None:
                hxtorch.init_hardware(spiking=True)
            else:
                hxtorch.init_hardware(
                    hxtorch.CalibrationPath(self.calibration_path))
        # Run
        results = self.execute(*args, **kwargs)

        # Release harrdware
        if not kwargs.get("mock"):
            hxtorch.release_hardware()

        return results

    # pylint: disable=too-many-arguments
    @abstractmethod
    def plot(self, path: str, data_mean: torch.Tensor, data: torch.Tensor,
             traces: torch.Tensor, title: str) -> None:
        """ Method to plot the data """
        fig = plt.figure(constrained_layout=True, figsize=(8, 5))
        fig.suptitle(title)

        axs = fig.subplots(nrows=1, ncols=2)
        axs[0].title.set_text("Example traces")
        axs[0].set(xlabel=r"$t$ [CADC Stamp]", ylabel=r"$v$ [CADC Value]")
        axs[0].plot(traces.numpy()[0], color="blue", alpha=0.3)
        axs[0].axhline(data_mean.numpy(), color="red", lw=2, label="Average")

        axs[1].title.set_text("Distribution (batch averaged)")
        axs[1].set(xlabel=r"$v$ [CADC Value]", ylabel=r"Count")
        axs[1].hist(data.numpy(), color="blue", bins=50)
        axs[1].axvline(data_mean.numpy(), color="red", lw=2, label="Average")
        plt.savefig(path)


class ReadoutExperiment(BaseExperiment):
    """ Specialization for experiments with non-spiking neurons """

    # pylint: disable=attribute-defined-outside-init
    def build_model(self, inputs: torch.Tensor, instance: snn.Instance,
                    transforms: Callable, **kwargs) -> snn.TensorHandle:
        """ Build model to map to hardware """
        # Layers
        synapses = snn.Synapse(
            self.input_size, self.output_size, instance=instance,
            transform=transforms)
        self.neurons = ReadoutNeuron(
            kwargs.get("shift_to_first", False), self.output_size,
            instance=instance, trace_scale=self.scale,
            trace_offset=self.offset, func=kwargs.get("func", F.LI),
            params=kwargs.get("params"))

        # In case of mock we want to initialize properly
        if kwargs.get("init_weight"):
            synapses.weight.data = kwargs.get("init_weight")(
                synapses.weight.data)

        # forward
        spikes = snn.NeuronHandle(spikes=inputs)
        currents = synapses(spikes)
        traces = self.neurons(currents)

        return traces


class SpikingExperiment(BaseExperiment):
    """ Specialization for experiments with spiking neurons """

    # pylint: disable=attribute-defined-outside-init
    def build_model(self, inputs: torch.Tensor, instance: snn.Instance,
                    transforms: Callable, **kwargs) -> snn.TensorHandle:
        """ Build model to map to hardware """
        # Layers
        synapses = snn.Synapse(
            self.input_size, self.output_size, instance=instance,
            transform=transforms)
        self.neurons = SpikingNeuron(
            kwargs.get("shift_to_first", False), self.output_size,
            instance=instance, trace_scale=self.scale,
            trace_offset=self.offset, func=kwargs.get("func", F.LIF),
            enable_spike_recording=kwargs.get("enable_spike_recording", False))

        # forward
        inputs = snn.NeuronHandle(spikes=inputs)
        currents = synapses(inputs)
        traces = self.neurons(currents)

        return traces


class MeasureBaseline(ReadoutExperiment):
    """ Measure baselines """

    def run(self, *args, **kwargs):
        """
        Execute experiment to measure hardware baselines. Returned data is
        given in units [CADC values].

        :returns: Returns a tuple (average baseline, average neuron-specific
                baseline, measured traces for each neuron,
                used hardware neurons).
        """
        # No events since we measure baselines
        inputs = torch.zeros(
            self.batch_size, self.time_length, self.input_size)
        traces = super().run(
            inputs, lambda x: torch.zeros_like(x), **kwargs)  # pylint: disable=unnecessary-lambda
        return traces

    # pylint: disable=attribute-defined-outside-init
    def post_process(self, traces: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """ post-process data """
        self.logger.INFO("Post process baselines...")
        # To numpy
        self.traces = traces.v_cadc.detach()
        # Extract baselines
        self.baselines = self.traces.mean(1).mean(0)
        # Baselines means
        self.baselines_mean = self.baselines.mean()
        return self.baselines_mean, self.baselines, self.traces, \
            self.hw_neurons

    # pylint: disable=arguments-differ
    def plot(self, path: str) -> None:
        """ Plot data """
        super().plot(
            path, self.baselines_mean, self.baselines, self.traces,
            "Baselines")


class MeasureUpperBoundary(ReadoutExperiment):
    """ Measure upper boundary """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = 5

    def run(self, *args, **kwargs):
        """
        Execute experiment to measure upper CADC boundaties. Returned data is
        given in units [CADC values].

        :returns: Returns a tuple (average upper boundary, average neuron-
                specific upper boundary, measured traces for each neuron,
                used hardware neurons).
        """
        # Events to push towards upper boundary
        inputs = torch.ones(
            self.batch_size, self.time_length, self.input_size)
        traces = super().run(
            inputs, lambda x: torch.zeros_like(x).fill_(63), **kwargs)
        return traces

    # pylint: disable=attribute-defined-outside-init
    def post_process(self, traces: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """ post-process data """
        self.logger.INFO("Post process upper boundaries...")
        # To numpy
        self.traces = traces.v_cadc.detach()
        # Extract upper boundarry
        self.upper = self.traces[:, 10:].mean(1).mean(0)
        # Upper mean
        self.upper_mean = self.upper.mean()
        return self.upper_mean, self.upper, self.traces, self.hw_neurons

    # pylint: disable=arguments-differ
    def plot(self, path: str) -> None:
        """ Plot data """
        super().plot(
            path, self.upper_mean, self.upper, self.traces, "Upper Boundaries")


class MeasureLowerBoundary(ReadoutExperiment):
    """ Measure lower boundary """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_size = 5

    def run(self, *args, **kwargs):
        """
        Execute experiment to measure lower CADC boundaties. Returned data is
        given in units [CADC values].

        :returns: Returns a tuple (average lower boundary, average neuron-
                specific lower boundary, measured traces for each neuron,
                used hardware neurons).
        """
        # Events to push towards upper boundary
        inputs = torch.ones(self.batch_size, self.time_length, self.input_size)
        traces = super().run(
            inputs, lambda x: torch.zeros_like(x).fill_(-63), **kwargs)
        return traces

    # pylint: disable=attribute-defined-outside-init
    def post_process(self, traces: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """ post-process data """
        self.logger.INFO("Post process lower boundaries...")
        # To numpy
        self.traces = traces.v_cadc.detach()
        # Extract upper boundarry
        self.lower = self.traces[:, 10:].mean(1).mean(0)
        # Upper mean
        self.lower_mean = self.lower.mean()
        return self.lower_mean, self.lower, self.traces, self.hw_neurons

    # pylint: disable=arguments-differ
    def plot(self, path: str) -> None:
        """ Plot data """
        super().plot(
            path, self.lower_mean, self.lower, self.traces, "Lower Boundaries")


class MeasureThreshold(SpikingExperiment):
    """ Measure thresholds """

    # pylint: disable=arguments-differ
    def run(self, **kwargs):
        """
        Execute experiment to measure thresholds. Returned data is given in
        units [CADC values].

        :returns: Returns a tuple (average threshold, average neuron-
                specific threshold, measured traces for each neuron,
                used hardware neurons).
        """
        # No spikes since we use constant current
        inputs = torch.zeros(
            self.batch_size, self.time_length, self.input_size)
        return super().run(
            inputs, lambda x: torch.zeros_like(x).fill_(63),
            enable_constant_current=True, **kwargs)

    # pylint: disable=attribute-defined-outside-init
    def post_process(self, traces: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """ post-process data """
        self.logger.INFO("Post process thresholds...")
        # To numpy
        self.traces = traces.v_cadc.detach()
        # Extract thresholds
        self.thresholds = torch.max(self.traces, 1)[0].mean(0)
        # Threshold means
        self.thresholds_mean = self.thresholds.mean()
        return self.thresholds_mean, self.thresholds, self.traces, \
            self.hw_neurons

    # pylint: disable=arguments-differ
    def plot(self, path: str) -> None:
        """ Plot data """
        super().plot(
            path, self.thresholds_mean, self.thresholds, self.traces,
            "Thresholds")


class MeasureWeightScaling(ReadoutExperiment):
    """ Measure weight scaling """

    max_weight = lola.SynapseWeightMatrix.Value.max
    min_weight = -lola.SynapseWeightMatrix.Value.max

    # pylint: disable=arguments-differ, attribute-defined-outside-init
    def execute(self, weight: int, **kwargs) -> torch.Tensor:
        # Keep weight for post-process
        self._current_weight = weight
        # Inputs
        inputs = torch.zeros(
            self.batch_size, self.time_length, self.input_size)
        inputs[:, 10, :] = 1
        # No spikes since we usw constant current
        return super().execute(
            inputs, lambda x: torch.zeros_like(x).fill_(weight), **kwargs)

    # pylint: disable=attribute-defined-outside-init
    def post_process(self, traces: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """ post-process data """
        # To numpy
        self.traces = traces.v_cadc.detach()

        # Get max/min PSP over time
        if self._current_weight >= 0:
            self.amp = self.traces.max(1)[0].mean(0)
        else:
            self.amp = self.traces.min(1)[0].mean(0)

        self.amp_mean = self.amp.mean()
        return self.amp_mean, self.amp, self.traces

    # pylint: disable=arguments-differ, too-many-arguments, too-many-locals
    def run(self, params: Union[F.LIParams, F.LIFParams],
            func_mock: Callable = F.li_integration,
            offset: Optional[torch.Tensor] = "measure",
            scale: Optional[torch.Tensor] = None, weight_step: int = 1,
            **kwargs) -> Tuple[torch.Tensor, ...]:
        """
        Measure scaling between mock and hardware weights.

        :param params: The LIF parameters used in mock mode.
        :param func_mock: The LI integration function used in mock mode.
            Note: Use the LI integration function here with the same parameters
                  as the LIF neurons. The LI function to avoid spiking
                  behavior.
        :param offset: The assumed hardware membrane offsets with which the
            neurons are shifted. This might either be of type float,
            torch.Tensor or a string with value 'measure' (default). If a float
            is given, the offset is applied to all neurons. If a torch.Tensor
            is given, it must hold one offset for each neuron. In case of
            string 'measure' the offsets are measured implicitly. If offset is
            None, no offset will be used at all.
        :param scale: The scale with which the hardware neuron traces are
            scaled. If None then the scale is measured an assumed that the
            neurons are supposed to spike at 1. If scale is a float, all traces
            are scaled with the same scale. If scale is a troch.Tensor if must
            hold one scale for each neuron.
        :param weight_step: The step width with which to increment the hardware
            weights.

        :returns: Returns the a tuple (average scaling, neuron-specific
            scaling, corresponding hardware neurons).
        """
        # Measure baselines
        if isinstance(offset, float):
            self.offset = offset
        if isinstance(offset, torch.Tensor):
            assert offset.shape[0] == self.output_size
            self.offset = offset
        if offset == "measure":
            self.logger.INFO("Measure baselines as offset...")
            handler = MeasureBaseline(self.calibration_path)
            _, self.offset, _, _ = handler.run()

        # Measure thresholds
        if scale is None:
            self.logger.INFO("Measure thresholds...")
            handler = MeasureThreshold(self.calibration_path)
            _, threshold, _, _ = handler.run()
            self.scale = 1. / (threshold - self.offset)
        if isinstance(scale, float):
            self.scale = scale
        if isinstance(scale, torch.Tensor):
            assert scale.shape[0] == self.output_size
            self.scale = scale

        # Sweep weights
        self.logger.INFO(f"Using weight step: {weight_step}")
        self.hw_weights = torch.arange(
            self.min_weight, self.max_weight + 1, weight_step)

        # Hardware amplitudes
        self.hw_amps = torch.zeros(self.hw_weights.shape[0], self.output_size)

        # Initialize hardware
        if self.calibration_path is None:
            hxtorch.init_hardware(spiking=True)
        else:
            hxtorch.init_hardware(
                hxtorch.CalibrationPath(self.calibration_path))

        # Sweep
        self.logger.INFO("Begin hardware weight sweep...")
        pbar = tqdm(total=self.hw_weights.shape[0])
        for i, weight in enumerate(self.hw_weights):
            # Measure
            _, self.hw_amps[i], _ = self.execute(weight, **kwargs)
            # Update
            pbar.set_postfix(
                weight=f"{weight}", mean_amp=self.hw_amps[i].mean())
            pbar.update()
        self.logger.INFO("Hardware weight sweep finished.")

        # Release hardware
        hxtorch.release_hardware()

        # Fit
        self.logger.INFO("Fit hardware data...")
        self.hw_scales = torch.zeros(self.output_size)
        for nrn in range(self.output_size):
            popt, _ = curve_fit(
                f=lambda x, a: a * x, xdata=self.hw_weights.numpy(),
                ydata=self.hw_amps[:, nrn].numpy())
            self.hw_scales[nrn] = popt[0]

        # Mock values
        self.logger.INFO("Begin mock weight sweep...")
        self.output_size = 1
        self.sw_weights = torch.arange(-1, 1 + 0.1, 0.1)
        # Hardware amplitudes
        self.sw_amps = torch.zeros(self.sw_weights.shape[0], 1)
        for i, weight in enumerate(self.sw_weights):
            # Measure
            _, self.sw_amps[i], _ = self.execute(
                weight, params=params, func=func_mock, mock=True,
                dt=params.dt,
                init_weight=lambda x: torch.zeros_like(x).fill_(weight))

        # SW scale
        popt, _ = curve_fit(
            f=lambda x, a: a * x, xdata=self.sw_weights.numpy(),
            ydata=self.sw_amps.reshape(-1).numpy())
        self.sw_scale = popt[0]

        # Resulting scales
        self.scales = self.sw_scale / self.hw_scales

        self.logger.INFO(
            f"Mock scale: {self.sw_scale}, HW scale: {self.hw_scales.mean()}"
            + f" +- {self.hw_scales.std()}")
        self.logger.INFO(f"SW -> HW translation factor: {self.scales.mean()}")

        return self.scales.mean(), self.scales, self.hw_neurons

    # pylint: disable=arguments-differ
    def plot(self, path: str):
        """ Plot measured data """
        fig = plt.figure(constrained_layout=True, figsize=(8, 5))
        axs = fig.subplots(nrows=1, ncols=2)

        # Function used for fitting
        # pylint: disable=invalid-name
        def linear(x, a):
            return a * x

        # Plot relation between SW PSP and HW PSP
        # HW
        axs[0].title.set_text(
            r"Scaling: $w_\mathrm{{hw}} = "
            + r"{0:.4} \cdot w_\mathrm{{sw}}$".format(self.scales.mean()))
        axs[0].set(
            xlabel=r"$w_\mathrm{hw}$ [a.u.]",
            ylabel=r"$\mathrm{max}_t(\mathrm{sign}(w)\cdot$"
            + r"$\mathrm{PSP})$ [a.u.]",
            xlim=[self.min_weight, self.max_weight])
        for nrn in range(self.hw_amps.shape[-1]):
            axs[0].scatter(
                self.hw_weights, self.hw_amps[:, nrn], color="orange",
                alpha=0.04, s=3)

        # Plot HW average
        x = np.linspace(self.min_weight, self.max_weight, 1000)
        scale_mean = linear(x, self.hw_scales.mean())
        scale_std = linear(x, self.hw_scales.std())
        axs[0].plot(
            x, scale_mean, lw=1, color="red",
            label=r"$\mathrm{{max}}_t(\mathrm{{sign}}(w_\mathrm{{hw}}) \cdot"
            + r"\mathrm{PSP}_\mathrm{{hw}}) = $"
            + r"$ {0:.3} \cdot w_\mathrm{{hw}}$".format(self.hw_scales.mean()))
        axs[0].fill_between(
            x, scale_mean + scale_std, scale_mean - scale_std, color="red",
            lw=0.5, alpha=0.3)

        # Syling
        axs[0].legend(loc="lower right")
        axs[0].xaxis.label.set_color("red")
        axs[0].spines["top"].set_edgecolor("red")
        axs[0].tick_params(axis="x", colors="red")

        # Plot SW PSP
        ax2 = axs[0].twiny()
        ax2.set(
            xlabel=r"$w_\mathrm{sw}$ [a.u.]", xlim=[-1., 1.])
        x = np.linspace(-1, 1, 1000)
        ax2.plot(
            x, linear(x, self.sw_scale), lw=1, color="blue",
            label=r"$\mathrm{{max}}_t(\mathrm{{sign}}(w_\mathrm{{sw}}) \cdot"
            + r"\mathrm{PSP}_\mathrm{{sw}}) = $"
            + r"$ {0:.3} \cdot w_\mathrm{{sw}}$".format(self.sw_scale))

        # Styling
        ax2.xaxis.label.set_color("blue")
        ax2.spines["top"].set_edgecolor("blue")
        ax2.tick_params(axis="x", colors="blue")
        ax2.legend(loc="upper left")

        # Hist
        axs[1].title.set_text(
            r'Distribution of $a^\mathrm{hw}$ with' + '\n'
            + r"$\mathrm{max}_t(\mathrm{sign}(w_\mathrm{hw})"
            + r"\cdot \mathrm{PSP}_\mathrm{hw}) = a^\mathrm{hw}"
            + r'\cdot w_\mathrm{hw}$')
        axs[1].hist(self.hw_scales.numpy(), bins=50, alpha=0.5, color="red")
        axs[1].axvline(
            self.hw_scales.mean(), color="black", label="Mean over neurons")
        axs[1].set(
            xlabel=r"$a^\mathrm{hw}$ [a.u.]", ylabel="Count")
        x = np.linspace(-1, 1, 1000)
        axs[1].legend()

        plt.savefig(path)


class MeasureTraceShift(BaseExperiment):
    """ Measure shift of cadc trace to spike train """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spike_time_in_us = 0

    # pylint: disable=attribute-defined-outside-init
    def build_model(self, inputs: torch.Tensor, instance: snn.Instance,
                    transforms: Callable, **kwargs) -> snn.TensorHandle:
        """ Build model to map to hardware """
        # Custom Neuron to return CADCHandle
        # pylint: disable=redefined-outer-name
        class ReadoutNeuron(snn.ReadoutNeuron):
            """ Readout neuron adjustment to return whole CADCHandle """

            # pylint: disable=no-member
            def post_process(self, hw_spikes, hw_cadc, hw_madc) \
                    -> Tuple[Optional[snn.CADCHandle], ...]:
                return hw_cadc, None

        # Layers
        synapses = snn.Synapse(
            self.input_size, self.output_size, instance=instance,
            transform=transforms)
        self.neurons = ReadoutNeuron(
            self.output_size, instance=instance, trace_scale=self.scale,
            trace_offset=self.offset, func=lambda x, hw_data: hw_data,
            params=kwargs.get("params"))

        # In case of mock we want to initialize properly
        if kwargs.get("init_weight"):
            synapses.weight.data = kwargs.get("init_weight")(
                synapses.weight.data)

        # forward
        spikes = snn.NeuronHandle(spikes=inputs)
        currents = synapses(spikes)
        traces = self.neurons(currents)

        return traces

    # pylint: disable=attribute-defined-outside-init, too-many-locals, arguments-differ
    def run(self, params: Union[F.LIParams, F.LIFParams], num_runs: int = 5,
            spike_time_in_us: int = 10, **kwargs) -> Tuple:
        """
        Measure the timeshift in us between CADC membrane readouts and spikes.

        :param params: The expected software LIF/LI params corresponding to the
            current hardware configuration.
        :param num_runs: The number of runs to perform. Each run executes a
            batch of size `batch_size` (can be alters in kwargs, default: 100).
            For plotting this should not be > 5.
        :param spike_time_in_us: The time the spike in us is send in. The
            resulting shift values are shifted by this value, such that this
            time only effects the actual spike time on hardware. Thus,
            resulting shift values assume this value to be 0.

        :returns: Returns a tuple (average time shift, time shifts, hardware
            neurons). Here, the average time shift is given in us and is
            computed by averaging over num_runs and batch_size. Time shifts is
            a torch tensor holding all measured time shifts, shape: (num_runs,
            batch_size, neurons). Hardware neurons gives the corresponding used
            hardware neurons.
        """
        # Measure baselines
        self.logger.INFO("Measure baselines as offset...")
        handler = MeasureBaseline(self.calibration_path)
        _, self.baselines, _, _ = handler.run()

        # inputs
        self.spike_time_in_us = spike_time_in_us
        inputs = torch.zeros(
            self.batch_size, self.time_length, self.input_size)
        inputs[:, self.spike_time_in_us, :] = 1.

        # fit function
        # TODO: fit function when having different time constants
        def fit_func(time, time_in, amp, tau_syn, v_init):
            return amp * np.greater(time - time_in, 0.) * (time - time_in) * \
                np.exp(- (time - time_in) / tau_syn) + v_init

        # loop over runs
        traceshifts = []
        self.logger.INFO("Begin hardware runs...")
        pbar = tqdm(total=num_runs)
        for _ in range(num_runs):
            # run
            traces, _ = super().run(
                inputs, lambda x: torch.zeros_like(x).fill_(63), **kwargs)
            raw_traces, times = traces.v_cadc.to_raw()
            raw_times_in_us = \
                times / float(hal.Timer.Value.fpga_clock_cycles_per_us)

            # loop batch
            shift_of_batches = []
            for batch in range(self.batch_size):
                # loop neurons
                shift_of_neurons = []
                for neuron in range(self.output_size):
                    # fit
                    try:
                        popt, _ = curve_fit(
                            fit_func,
                            raw_times_in_us[batch, :, neuron].numpy(),
                            raw_traces[batch, :, neuron].numpy(),
                            p0=[
                                self.spike_time_in_us, 5.,
                                1. / params.tau_syn_inv / 1e-6,
                                self.baselines[neuron]])
                        shift_of_neurons.append(
                            popt[0] - self.spike_time_in_us)
                    except RuntimeError:
                        shift_of_neurons.append(np.nan)
                shift_of_batches.append(np.array(shift_of_neurons))
            traceshifts.append(np.array(shift_of_batches))

            # Update
            pbar.set_postfix(
                batch_avg_shift=f"{np.nanmean(traceshifts[-1])}")
            pbar.update()

        # trace_shifts now has shape (num_runs, batch_size, output_size)
        self.traceshifts = np.array(traceshifts)
        self.avg_traceshift = self.traceshifts[
            ~np.isnan(self.traceshifts) & (self.traceshifts > -5)
            & (self.traceshifts < 1)].flatten().mean()

        return self.avg_traceshift, torch.tensor(self.traceshifts), \
            self.hw_neurons

    def post_process(self, traces: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """ post-process data """
        # To numpy
        return traces, self.hw_neurons

    # pylint: disable=arguments-differ
    def plot(self, path: str) -> None:
        """ Plot measurements """
        fig = plt.figure(constrained_layout=True, figsize=(20, 5))
        fig.suptitle(
            f"Shift between CADC and spikes: {self.avg_traceshift}")
        axs = fig.subplots(nrows=1, ncols=3)

        # assume that shifts are in between -5us to +1us
        # and other values should only be due to bad fits
        max_idx = 1
        min_idx = -5

        # distribution for single run and up to 5 batch entries
        colors = ["C0", "C1", "C2", "C3", "C4"]
        assert self.traceshifts.shape[-1] > 5
        axs[0].set_title(
            r"Distribution of $t^\mathrm{{fit}}_\mathrm{{in}}$ "
            + "for single run, batch_size = "
            + f"{min(5, self.batch_size):.0f} and "
            + f"{self.output_size:.0f} neurons")
        axs[0].set(xlabel=r"$t$ [us]", ylabel=r"Count")
        axs[0].axvline(0., c="black", ls="dashed", label=r"$t_\mathrm{{in}}$")

        # plot histogram for each batch entry
        for batch_entry in range(min(5, self.batch_size)):
            # select traceshifts
            select_shifts = self.traceshifts[0, batch_entry, :].flatten()
            select_shifts = select_shifts[
                ~np.isnan(select_shifts)
                & (select_shifts > min_idx) & (select_shifts < max_idx)]
            # plot
            axs[0].hist(select_shifts, bins=50,
                        color=colors[batch_entry],
                        alpha=0.5, label=r"$t^\mathrm{{fit}}_\mathrm{{in}}$"
                        + f", batch {batch_entry:.0f}")
            axs[0].axvline(select_shifts.mean(), color=colors[batch_entry],
                           lw=2, label=f"Avg. {select_shifts.mean():.2f}, "
                           + f"batch {batch_entry:.0f}")
        axs[0].legend()

        # distribution for batch_size=1 over multiple runs
        axs[1].set_title(
            r"Distribution of $t^\mathrm{{fit}}_\mathrm{{in}}$ "
            + "for batch 0, "
            + f"{self.traceshifts.shape[0]:.0f} runs and "
            + f"{self.output_size:.0f} neurons")
        axs[1].set(xlabel=r"$t$ [us]", ylabel=r"Count")
        axs[1].axvline(0., c="black", ls="dashed", label=r"$t_\mathrm{{in}}$")
        # select traceshifts
        select_shifts = self.traceshifts[:, 0, :].flatten()
        select_shifts = select_shifts[
            ~np.isnan(select_shifts) & (select_shifts > min_idx)
            & (select_shifts < max_idx)]
        # plot
        axs[1].hist(select_shifts, bins=50,
                    color="C0", alpha=0.5,
                    label=r"$t^\mathrm{{fit}}_\mathrm{{in}}$",
                    histtype="stepfilled")
        axs[1].axvline(select_shifts.mean(), color="C0", lw=2,
                       label=f"Avg., {select_shifts.mean():.2f}")
        axs[1].legend()

        # complete distribution
        axs[2].set_title(
            r"Distribution of $t^\mathrm{{fit}}_\mathrm{{in}}$ "
            + f"for batch_size = {self.batch_size}, "
            + f"{self.traceshifts.shape[0]:.0f} runs and "
            + f"{self.output_size:.0f} neurons")
        axs[2].set(xlabel=r"$t$ [us]", ylabel=r"Count")
        axs[2].axvline(0., c="black", ls="dashed", label=r"$t_\mathrm{{in}}$")
        # select traceshifts
        select_shifts = self.traceshifts[
            ~np.isnan(self.traceshifts) & (self.traceshifts > min_idx)
            & (self.traceshifts < max_idx)].flatten()
        # plot
        axs[2].hist(select_shifts, bins=50,
                    color="C0", alpha=0.5,
                    label=r"$t^\mathrm{{fit}}_\mathrm{{in}}$",
                    histtype="stepfilled")
        axs[2].axvline(select_shifts.mean(), color="C0", lw=2,
                       label=f"Average, {select_shifts.mean():.2f}")
        axs[2].legend()

        # save
        self.logger.INFO(f"Saving plot to '{path}'")
        plt.savefig(path)
