"""
Test spiking neuron modules
"""
import unittest

import torch

import hxtorch
from hxtorch import spiking as hxsnn


hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)
logger = hxtorch.logger.get("hxtorch.test.sw.test_spiking_modules")


class TestAELIF(unittest.TestCase):
    """ Test AELIF """

    def test_ignore_tau_mem(self):
        """ Tests whether tau_mem is ignored if c_m and g_l are set """
        spikes = torch.zeros((50, 1, 1))
        spikes[5] = 1
        spike_handle = hxsnn.LIFObservables(spikes=spikes)

        experiment_set = hxsnn.Experiment(mock=True)

        syn_set = hxsnn.Synapse(1, 1, experiment_set)
        syn_set.weight.data = torch.ones_like(
                syn_set.weight.data
        )
        neuron_g_set = hxsnn.AELIF(
                1,
                experiment_set,
                tau_mem=0.,
                tau_syn=10e-6,
                membrane_capacitance=10e-6,
                leak_conductance=1.
        )

        experiment_not_set = hxsnn.Experiment(mock=True)
        syn_not_set = hxsnn.Synapse(1, 1, experiment_not_set)
        syn_not_set.weight.data = torch.ones_like(
                syn_not_set.weight.data
        )
        neuron_g_not_set = hxsnn.AELIF(
                1,
                experiment_not_set,
                tau_mem=10e-6,
                tau_syn=10e-6,
                membrane_capacitance=10e-6,
        )

        out_set = neuron_g_set(syn_set(spike_handle))
        hxsnn.run(experiment_set, spike_handle.spikes.shape[0])
        out_not_set = neuron_g_not_set(syn_not_set(spike_handle))
        hxsnn.run(experiment_not_set, spike_handle.spikes.shape[0])

        self.assertTrue(
            torch.allclose(out_set.membrane_cadc, out_not_set.membrane_cadc)
        )


if __name__ == "__main__":
    unittest.main()
