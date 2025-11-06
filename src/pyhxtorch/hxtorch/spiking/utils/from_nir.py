'''
Translate a NIRGraph to an hxtorch SNN.
'''

from dataclasses import dataclass, field
from functools import partial
import torch
import nir

from hxtorch.spiking.experiment import Experiment
from hxtorch.spiking.handle import LIFObservables
from hxtorch.spiking.modules import AELIF, LIF, LI, InputNeuron, Synapse
from hxtorch.spiking.parameter import (HXTransformedModelParameter,
                                       MixedHXModelParameter)
from hxtorch.spiking.run import run
from hxtorch.spiking.transforms import weight_transforms


@dataclass
class ConversionConfig:
    '''
    Configuration for the conversion of NIRGraph to hxtorch SNN.

    Some parameters are a dict with node names as keys such that the
    parametrization for each layer can be set individually.
    '''
    dt: float = 1e-6  # pylint: disable=invalid-name
    calib_path: str = None
    weight_scale: float = 64.
    trace_scale: float = 1. / 50.
    trace_shift: float = 0e-6
    cadc_recording: bool = True  # no training if False
    mock: bool = True
    input_loopback: bool = True
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


def _get_keys(graph, node_class):
    '''
    Return array of keys of nodes of node class
    '''
    key_array = []
    for key, node in graph.nodes.items():
        if isinstance(node, node_class):
            key_array.append(key)

    return key_array


def _map_nir_to_hxtorch(
    exp: Experiment,
    node: nir.NIRNode,
    cfg: ConversionConfig,
) -> torch.nn.Module:

    if isinstance(node, nir.Linear):
        module = Synapse(
            in_features=node.weight.shape[1],
            out_features=node.weight.shape[0],
            experiment=exp,
            transform=partial(
                weight_transforms.linear_saturating, scale=cfg.weight_scale))
        module.weight.data = torch.Tensor(node.weight)
        return module
    if isinstance(node, nir.CubaLI):
        size = node.input_type["input"][0]
        for param in [node.tau_mem, node.tau_syn, node.v_leak]:
            if not all(x == param[0] for x in param):
                raise ValueError(
                    "CubaLI parameters must be homogeneous across neurons"
                    "in a layer.")

        tau_mem = node.tau_mem[0] * 1e-3  # convert ms to s
        tau_syn = node.tau_syn[0] * 1e-3  # convert ms to s
        leak = node.v_leak[0]

        module = LI(
            size=size,
            experiment=exp,
            leak=MixedHXModelParameter(leak, 80),
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            trace_scale=cfg.trace_scale,
            cadc_time_shift=cfg.trace_shift,
            shift_cadc_to_first=True
        )
        return module
    if isinstance(node, nir.CubaLIF):
        size = node.input_type["input"][0]
        for param in [node.tau_mem, node.tau_syn, node.v_leak, node.v_reset,
                      node.v_threshold, node.r]:
            if not all(x == param[0] for x in param):
                raise ValueError(
                    "CubaLIF parameters must be homogeneous across neurons"
                    "in a layer.")

        tau_mem = node.tau_mem[0] * 1e-3  # convert ms to s
        tau_syn = node.tau_syn[0] * 1e-3  # convert ms to s
        leak = node.v_leak[0]
        reset = node.v_reset[0]
        threshold = node.v_threshold[0]
        r = node.r[0]  # pylint: disable=invalid-name

        module = LIF(
            size,
            experiment=exp,
            leak=MixedHXModelParameter(leak, 80),
            reset=MixedHXModelParameter(reset, 80),
            threshold=MixedHXModelParameter(threshold, 150),
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            membrane_capacitance=(
                HXTransformedModelParameter(
                    tau_mem / r,
                    lambda model_value: model_value / tau_mem * r * 63)),
            trace_scale=cfg.trace_scale,
            cadc_time_shift=cfg.trace_shift,
            shift_cadc_to_first=True,
            enable_cadc_recording=cfg.cadc_recording)
        return module
    if isinstance(node, nir.Input):
        size = node.input_type["input"][0]
        module = InputNeuron(
            size,
            experiment=exp)
        return module
    raise NotImplementedError(
        f"Node type {type(node)} is not supported for conversion to \
            hxtorch.")


def from_nir(
    graph: nir.NIRGraph,
    cfg: ConversionConfig = None
) -> torch.nn.Module:
    """
    Converts a NIRGraph to an hxtorch module.

    :Limitations:
    - Only NIRGraphs with exactly one Input and one Output node
    - Only Linear, CubaLI, and CubaLIF nodes are supported
    """

    class SNN(torch.nn.Module):
        def __init__(self, cfg: ConversionConfig):
            super().__init__()
            # Experiment instance to work on
            self.exp = Experiment(mock=cfg.mock, dt=cfg.dt)
            if not cfg.mock:
                if cfg.calib_path is not None:
                    self.exp.default_execution_instance.load_calib(
                        cfg.calib_path)
                self.exp.default_execution_instance.input_loopback = \
                    cfg.input_loopback

            # Build dict of hxtorch modules
            self.hxnodes = torch.nn.ModuleDict()
            self.spike_keys = []
            self.output = {}
            for node_key in graph.nodes:
                if isinstance(graph.nodes[node_key], nir.Output):
                    continue
                node = graph.nodes[node_key]
                self.hxnodes[node_key] = _map_nir_to_hxtorch(self.exp, node,
                                                             cfg)
                if isinstance(self.hxnodes[node_key], (AELIF,
                                                       InputNeuron)):
                    if not (isinstance(self.hxnodes[node_key], AELIF)
                            and not self.hxnodes[node_key].fire):
                        self.spike_keys.append(node_key)

            # Device
            self.device = cfg.device
            self.to(self.device)

        def forward(self, spikes_dict):
            """
            Forward pass through the SNN.

            :param spikes_dict: Dictionary of input spike tensors with keys \
                as node names.
            :return: Dictionary of spike tensors for each node.
            """
            input_node = _get_keys(graph, nir.Input)[0]
            output_node = _get_keys(graph, nir.Output)[0]

            for edge in graph.edges:
                in_node = edge[0]
                out_node = edge[1]

                if in_node == input_node:
                    spikes = spikes_dict[in_node].to(self.device)
                    spikes_handle = LIFObservables(spikes=spikes)
                    self.output[in_node] = self.hxnodes[in_node](spikes_handle)
                    self.output[out_node] = \
                        self.hxnodes[out_node](self.output[in_node])
                elif out_node == output_node:
                    pass
                else:
                    self.output[out_node] = \
                        self.hxnodes[out_node](self.output[in_node])

            # Execute on hardware
            run(self.exp, spikes.shape[0])

            return {key: value.spikes
                    for key, value in self.output.items()
                    if hasattr(value, 'spikes')}

    if cfg is None:
        # Standard ConversionConfig is generated
        cfg = ConversionConfig()

    return SNN(cfg)
