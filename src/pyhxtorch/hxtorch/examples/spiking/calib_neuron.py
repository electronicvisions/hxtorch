""" """
import torch
import hxtorch
import hxtorch.core as hxcore
import hxtorch.spiking as hxsnn


def main():
    log = hxtorch.logger.get("hxtorch.examples.spiking.calib_neuron")

    # Initialize hardware
    hxtorch.init_hardware()

    # Experiment
    experiment = hxsnn.Experiment(mock=False)

    # Modules
    synapse = hxsnn.Synapse(
        in_features=2, out_features=4, experiment=experiment)
    neuron = hxsnn.LIF(
        size=4,
        experiment=experiment,
        tau_syn=hxcore.HXParameter(6e-6),
        tau_mem=hxcore.HXParameter(20e-6),
        refractory_time=hxcore.HXParameter(1e-6),
        leak=hxcore.HXParameter(80.),
        reset=hxcore.HXParameter(80.),
        threshold=hxcore.HXParameter(torch.tensor([80., 90., 100., 120.])))

    # Weights
    torch.nn.init.normal_(synapse.weight, mean=63., std=0.)

    # Input
    spikes = torch.zeros((20, 1, synapse.in_features))
    spikes[5] = 1.

    # Forward
    spike_handle = hxsnn.LIFObservables(spikes=spikes)
    output = neuron(synapse(spike_handle))

    # Execute
    hxsnn.run(experiment=experiment, runtime=spikes.shape[0])

    # Print spike output. Neuron 0 should spike more often than neuron 1.
    # Neurons 2 and 3 should emit no spikes.
    log.INFO("Spikes: ", output.spikes.to_sparse())

    # Calibration results are stored to be accessed by user
    log.INFO("Neuron params: ", neuron.params_dict())

    # Modify parameters, e.g. leak over threshold
    neuron.leak.hardware_value = 1022.

    # Zero input
    spikes[:, :, :] = 0.

    # Forward
    spike_handle = hxsnn.LIFObservables(spikes=spikes)
    output = neuron(synapse(spike_handle))
    # Execute
    hxsnn.run(experiment=experiment, runtime=spikes.shape[0])

    log.INFO("Spikes: ", output.spikes.to_sparse())

    # Release
    hxtorch.release_hardware()

    # Return observables
    return output.spikes, output.membrane_cadc


if __name__ == "__main__":
    for key in ["hxcomm", "grenade", "stadls", "calix"]:
        other_logger = hxtorch.logger.get(key)
        hxtorch.logger.set_loglevel(other_logger, hxtorch.logger.LogLevel.WARN)
    spiketrains, traces = main()
