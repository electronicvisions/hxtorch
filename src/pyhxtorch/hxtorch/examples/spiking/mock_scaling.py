"""
Example for usage of utils.measure_mock_scaling with plots
"""
from typing import Dict
import argparse
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch

import hxtorch
import hxtorch.spiking as hxsnn
from hxtorch.spiking.utils.dynamic_range.boundary import get_dynamic_range
from hxtorch.spiking.utils.dynamic_range.weight_scaling import get_weight_scaling
from hxtorch.spiking.utils.dynamic_range.threshold import get_trace_scaling
from hxtorch.spiking.transforms.weight_transforms import linear_saturating

log = hxtorch.logger.get("hxtorch.examples.spiking.mock_scaling")
hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)


def get_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser with all the options.
    """
    parser = argparse.ArgumentParser(
        description="hxtorch measure mock scaling example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--weight-step", type=int, default=10)
    parser.add_argument("--tau-syn", type=float, default=10e-6)
    parser.add_argument("--tau-mem", type=float, default=10e-6)
    parser.add_argument("--model-threshold", type=float, default=1.0)
    parser.add_argument("--model-leak", type=float, default=0.0)
    parser.add_argument("--model-reset", type=float, default=0.0)
    parser.add_argument("--bss2-threshold", type=float, default=125)
    parser.add_argument("--bss2-leak", type=float, default=80)
    parser.add_argument("--bss2-reset", type=float, default=80)
    parser.add_argument(
        "--calibration-path", type=str, metavar="<path>", default=None,
        help="path to custom calibration to use instead of latest nightly")
    parser.add_argument("--plot-path", type=str, default="./mock_scaling.png")

    return parser


def plot_scaled_trace(args, inputs, bss2_traces, mock_trace):
    _, axs = plt.subplots(nrows=1, sharex="col", figsize=(6, 3))
    input_events = torch.nonzero(inputs)
    axs.vlines(input_events[:, 0], ymin=-1, ymax=2, color="orange", label="Inputs")
    axs.set_ylabel(r"$v_m^\mathrm{CADC}$ [CADC Value]")
    axs.plot(mock_trace, color="red")
    axs.plot(np.stack(bss2_traces).T, color="blue", alpha=0.2)
    axs.plot(np.stack(bss2_traces).mean(0), color="blue")
    axs.set_ylim(-0.2, 0.6)
    plt.savefig(args.plot_path, dpi=300)


def run(inputs: torch.Tensor, nrn_params: Dict[str, hxsnn.parameter.HXBaseParameter],
        calib_path: str = None, mock=False, weight_scale=63., trace_offset=0.,
        trace_scale=1., n_runs=10):
    traces = []

    hxtorch.init_hardware()
    for _ in range(n_runs):
        # Experiment
        exp = hxsnn.Experiment(dt=1e-6, mock=mock)
        # Modules
        syn = hxsnn.Synapse(
            in_features=1,
            out_features=1,
            experiment=exp,
            transform=partial(linear_saturating, scale=weight_scale))
        lif = hxsnn.LIF(
            size=1,
            **nrn_params,
            experiment=exp,
            cadc_time_shift=-1,
            trace_offset=trace_offset,
            trace_scale=trace_scale)
        syn.weight.data.fill_(1.)

        if calib_path is not None and not mock:
            exp.default_execution_instance.load_calib(calib_path)

        # Forward
        g = syn(hxsnn.LIFObservables(spikes=inputs))
        z = lif(g)
        hxsnn.run(exp, 50)  # dt
        traces.append(z.membrane_cadc.detach().numpy().reshape(-1))
    hxtorch.release_hardware()
    return traces


def main(args: argparse.Namespace):
    """
    Returns and creates plots of the measured mock scaling given a calibration
    and software settings
    """
    lif_params = {
        "tau_mem": hxsnn.MixedHXModelParameter(args.tau_mem, args.tau_mem),
        "tau_syn": hxsnn.MixedHXModelParameter(args.tau_syn, args.tau_syn),
        "leak": hxsnn.MixedHXModelParameter(args.model_leak, args.bss2_leak),
        "reset": hxsnn.MixedHXModelParameter(args.model_reset, args.bss2_reset),
        "threshold": hxsnn.MixedHXModelParameter(
            args.model_threshold, args.bss2_threshold)}

    # Measure baseline
    baselines, _, _ = get_dynamic_range(calib_path=args.calibration_path,
                                        params=lif_params)
    log.INFO(f"CADC membrane baseline: {baselines}")

    # Measure hardware <-> model trace scaling
    trace_scale = get_trace_scaling(calib_path=args.calibration_path,
                                    params=lif_params)
    log.INFO(f"CADC trace scaling: {trace_scale}")

    # Next we tune the weight_scaling such that the PSP in the software model
    # and on hardware look the same
    weight_scaling = get_weight_scaling(
        params=lif_params,
        calib_path=args.calibration_path,
        weight_step=args.weight_step)

    # Measure some traces for comparison
    inputs = torch.zeros((50, 1, 1))
    inputs[10] = 1
    mock_trace = run(inputs, lif_params,
                     calib_path=args.calibration_path, mock=True, n_runs=1)[0]
    bss2_traces = run(
        inputs, lif_params, calib_path=args.calibration_path,
        trace_offset=baselines.item(), trace_scale=trace_scale.item(),
        weight_scale=weight_scaling.item())

    # Display
    plot_scaled_trace(args, inputs, bss2_traces, mock_trace)


if __name__ == "__main__":
    main(get_parser().parse_args())
