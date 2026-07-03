from dlens_vx_v3 import hal


class ReadoutSource():
    CURRENT_EXC = hal.NeuronConfig.ReadoutSource.exc_synin
    CURRENT_INH = hal.NeuronConfig.ReadoutSource.inh_synin
    VOLTAGE = hal.NeuronConfig.ReadoutSource.membrane
    ADAPTATION = hal.NeuronConfig.ReadoutSource.adaptation
