"""
Translate spikes from a jaxsnn spike representation to NIRGraphData.
"""

import numpy as np
from nir.data_ir import NIRGraphData, NIRNodeData, TimeGriddedData


def to_nir_data(hxtorch_dict: dict, hxtorch_model) -> NIRGraphData:
    '''
    hxtorch data comes as torch tensor of size (n_time_steps, batch_size,
    n_neurons)
    '''
    nir_nodes = {}

    for key, spikes in hxtorch_dict.items():
        spikes = spikes.detach().cpu().numpy()
        spikes = np.moveaxis(spikes, 1, 0)

        nir_node_data = NIRNodeData(
            {'spikes': TimeGriddedData(spikes, hxtorch_model.exp.dt)})
        nir_nodes[key] = nir_node_data

    nir_data = NIRGraphData(nir_nodes)
    return nir_data
