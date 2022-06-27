#include <vector>
#include <torch/torch.h>


namespace hxtorch::snn {

/**
 * Convert a torch tensor of spikes with a dense (but discrete) time representation into spike
 * times.
 *
 * @param times A tensor of shape (batch_size, time_length, population_size) holding spike
 * represemted as ones. Absent spikes are represented by zeros.
 * @param dt The temporal resolution of the spike tensor.
 * @return A vector with the first dimension being the batch dimension and the second dimension the
 * neuron index, holding a list of spike times of the corresonding neuron, i.e. shape (batch,
 * neuron index, spike times).
 */
std::vector<std::vector<std::vector<float>>> tensor_to_spike_times(torch::Tensor times, float dt);

}
