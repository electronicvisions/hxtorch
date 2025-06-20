"""
Translate an hxtorch SNN to a NIRGraph.
"""

import numpy as np
import torch
import nir
from hxtorch.spiking.experiment import Experiment
from hxtorch.spiking.modules import AELIF, Synapse
from hxtorch.spiking.run import run


def _map_hxtorch_to_nir(module):
    if isinstance(module, Synapse):
        weight = module.weight.detach().numpy()
        return nir.Linear(weight)
    if isinstance(module, AELIF):

        if module.fire:
            size = module.size
            tau_mem = module.tau_mem.model_value
            tau_syn = module.tau_syn.model_value
            v_leak = module.leak.model_value
            v_reset = module.reset.model_value
            v_threshold = module.threshold.model_value
            return nir.CubaLIF(tau_mem=np.array(size * [tau_mem]),
                               tau_syn=np.array(size * [tau_syn]),
                               r=np.array(size * [1.]),
                               v_leak=np.array(size * [v_leak]),
                               v_reset=np.array(size * [v_reset]),
                               v_threshold=np.array(size * [v_threshold]))
        size = module.size
        tau_mem = module.tau_mem.model_value
        tau_syn = module.tau_syn.model_value
        v_leak = module.leak.model_value
        return nir.CubaLI(tau_mem=np.array(size * [tau_mem]),
                          tau_syn=np.array(size * [tau_syn]),
                          r=np.array(size * [1.]),
                          v_leak=np.array(size * [v_leak]))
    raise NotImplementedError(
        f"Conversion of module {module} to NIR is not implemented.")


class SNN(torch.nn.Module):
    """
    Base SNN class for to-NIR conversion. It is necessary to define the forward
    pass.
    """
    # pylint: disable=invalid-name

    def __init__(self, dt: float = 1.0e-6, mock: bool = True,
                 device: torch.device = torch.device("cpu")) -> None:
        """
        Initialize the SNN.

        :param dt: Time-binning width.
        :param mock: Indicating whether to train in software (True) or on
            hardware (False).
        """
        super().__init__()

        self.exp = Experiment(mock=mock, dt=dt)

        # Device
        self.device = device
        self.to(device)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:  # pylint: disable=unused-argument, useless-return
        """
        Perform a forward pass. To use the SNN class for to-NIR conversion, it
        is necessary to define the forward pass.

        :param spikes: torch.Tensor holding spikes as input.

        :return: Returns the output of the network, i.e. membrane traces of the
            readout neurons.
        """

        run(self.exp, spikes.shape[0])
        return None


def to_nir(snn: SNN, input_sample) -> nir.NIRGraph:
    """
    Convert a hxtorch SNN to a NIR graph.

    :param snn: The hxtorch SNN to convert, where snn.exp is the experiment
        object. Furthermore the SNN must use modules that are convertible to
        NIR (e.g. Synapse, AELIF).
    :param input_sample: A single input sample to the SNN.
    """

    _ = snn(input_sample)

    input_key = 0
    output_key = 0

    nir_nodes = {}
    nir_edges = []

    graph = snn.exp.modules.prev_graph

    for node in graph.nodes():
        if not list(graph.predecessors(node)):
            input_key = str(node)
            nir_nodes[str(node)] = nir.Input(input_type=np.array([1]))
        elif not list(graph.successors(node)):
            output_key = str(node)
            nir_nodes[str(node)] = nir.Output(output_type=np.array([1]))
            nir_edges += tuple((str(pred), str(node)) for pred in
                               graph.predecessors(node))
        else:
            hxmodule = [module for module, key in
                        snn.exp.modules.nodes.items() if key == node]
            name = str(list(snn.exp.modules.nodes.keys()).index(hxmodule[0]))
            nir_nodes[name] = _map_hxtorch_to_nir(hxmodule[0])
            nir_edges += tuple((str(pred), str(node)) for pred in
                               graph.predecessors(node))

    # set correct input/output types
    next_node = [post_node for (pre_node, post_node) in nir_edges
                 if pre_node == input_key]
    input_size = nir_nodes[next_node[0]].input_type['input']
    nir_nodes[input_key] = nir.Input(input_type=np.array(input_size))

    prev_node = [pre_node for (pre_node, post_node) in nir_edges
                 if post_node == output_key]
    output_size = nir_nodes[prev_node[0]].output_type['output']
    nir_nodes[output_key] = nir.Output(output_type=np.array(output_size))

    return nir.NIRGraph(nir_nodes, nir_edges)
