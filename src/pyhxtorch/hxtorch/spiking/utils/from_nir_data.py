"""
Translate NIRGraphData to a hxtorch-compatible spike representation.
"""

import torch
from nir.data_ir import EventData, NIRGraphData, NIRNodeData


def from_nir_data(nir_graph_data: NIRGraphData, hxtorch_model) -> dict:
    hxtorch_dict = {}

    for node_key, nir_node_data in nir_graph_data.nodes.items():
        if isinstance(nir_node_data, NIRNodeData):
            for observable, data in nir_node_data.observables.items():
                if observable == 'spikes':
                    if isinstance(data, EventData):
                        data = data.to_time_gridded(hxtorch_model.exp.dt)

                    hxtorch_spikes = torch.tensor(data.data)
                    hxtorch_spikes = hxtorch_spikes.permute(1, 0, 2)
                else:
                    raise NotImplementedError('Only spikes are supported as'
                                              'observables yet.')
        else:
            raise NotImplementedError('The translation of nested NIRGraphData'
                                      'is not supported.')

        hxtorch_dict[node_key] = hxtorch_spikes.float()

    return hxtorch_dict
