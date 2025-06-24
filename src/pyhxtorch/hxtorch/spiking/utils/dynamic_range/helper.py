""" Helper functions for dynamic ranges """
from dlens_vx_v3 import lola, halco

from hxtorch.spiking.modules import LIF, LI


class ConstantCurrentMixin:

    current_type: lola.AtomicNeuron.ConstantCurrent.Type \
        = lola.AtomicNeuron.ConstantCurrent.Type.source
    enable_current: bool = True

    def configure_hw_entity(self, neuron_id: int,
                            neuron_block: lola.NeuronBlock,
                            coord: halco.LogicalNeuronOnDLS) \
            -> lola.NeuronBlock:
        super().configure_hw_entity(neuron_id, neuron_block, coord)
        if self.enable_current:
            for nrn in halco.iter_all(halco.AtomicNeuronOnDLS):
                neuron_block.atomic_neurons[nrn] \
                    .constant_current.i_offset = 1000
                neuron_block.atomic_neurons[nrn] \
                    .constant_current.enable = True
                neuron_block.atomic_neurons[nrn] \
                    .constant_current.type = self.current_type
        return neuron_block


class ConstantCurrentNeuron(ConstantCurrentMixin, LIF):
    pass


class ConstantCurrentReadoutNeuron(ConstantCurrentMixin, LI):
    pass
