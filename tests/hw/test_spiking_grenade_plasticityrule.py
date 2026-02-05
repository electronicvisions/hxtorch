import textwrap
import unittest

import torch
from dlens_vx_v3 import hal, halco
import pygrenade_vx as grenade
import hxtorch
import hxtorch.snn as hxsnn
from hxtorch.spiking.utils import calib_helper


class TestSpikingGrenadePlasticityRule(unittest.TestCase):
    """ Test script for running plasticity rules from hxtorch"""

    def setUp(self):
        hxtorch.init_hardware()

    def tearDown(self):
        hxtorch.release_hardware()

    def _run_test(self, target_value, cadc_recording, plasticity_rule):
        # Experiment
        exp = hxsnn.Experiment(dt=1e-6)
        exp.default_execution_instance.load_calib(
            calib_helper.nightly_calix_native_path())

        # define PPU program symbols: init values and what to read back
        init_symbol = hal.PPUMemoryBlock(halco.PPUMemoryBlockSize(1))
        init_symbol.words = [hal.PPUMemoryWord(
            hal.PPUMemoryWord.Value(0x0))]
        exp.default_execution_instance.write_ppu_symbols = {
            "scalar_result": {
                halco.HemisphereOnDLS.top: init_symbol,
                halco.HemisphereOnDLS.bottom: init_symbol
            }
        }
        exp.default_execution_instance.read_ppu_symbols = {"scalar_result"}

        custom_plasticity_rule = grenade.network.PlasticityRule()
        val = grenade.network.PlasticityRule.Timer.Value
        custom_plasticity_rule.timer.start = val(1000)
        custom_plasticity_rule.timer.period = val(10000)
        custom_plasticity_rule.timer.num_periods = 1
        custom_plasticity_rule.kernel = textwrap.dedent("""
        #include "grenade/vx/ppu/neuron_view_handle.h"
        #include "grenade/vx/ppu/synapse_array_view_handle.h"
        #include "grenade/vx/ppu/time.h"
        #include <array>
        using namespace grenade::vx::ppu;
        volatile uint32_t scalar_result;

        void PLASTICITY_RULE_KERNEL(
        std::array<SynapseArrayViewHandle, 2>& synapses,
        std::array<NeuronViewHandle, 0>& /* neurons */)
        {{
            scalar_result = {val};
        }}
        """.format(val=target_value))

        # Modules
        if not plasticity_rule:
            custom_plasticity_rule = None
        syn = hxsnn.Synapse(
            in_features=1,
            out_features=1,
            experiment=exp,
            plasticity_rule=custom_plasticity_rule)

        lif = hxsnn.LIF(
            size=1,
            experiment=exp,
            enable_cadc_recording=cadc_recording,
            enable_madc_recording=True,
            enable_spike_recording=True,
            record_neuron_id=0,
            cadc_time_shift=-1)

        # Weights on hardware are between -63 to 63
        syn.weight.data.fill_(63)

        # Some random input spikes
        inputs = torch.zeros((50, 1, 1))
        inputs[[10, 15, 20, 30]] = 1  # in dt

        # Forward
        g = syn(hxsnn.LIFObservables(spikes=inputs))
        z = lif(g)

        hxsnn.run(exp, 50)  # dt

        return exp

    def test_readwrite_ppu_symbols(self):
        target_value = 123
        exp = self._run_test(target_value, False, True)
        for _, ppu_symbols_read in exp.ppu_symbols_read[0].items():
            self.assertEqual(int(ppu_symbols_read[
                grenade.common.ChipOnConnection(0)][
                'scalar_result'][
                halco.HemisphereOnDLS(0)].words[0].value), target_value)

    def test_cadc_and_plasticityrule(self):
        with self.assertRaises(ValueError):
            self._run_test(12345, True, True)

    def test_no_ppu_symbols_read(self):
        exp = self._run_test(12345, False, False)
        self.assertFalse(exp.ppu_symbols_read)


if __name__ == "__main__":
    unittest.main()
