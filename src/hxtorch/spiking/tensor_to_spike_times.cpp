#include "hxtorch/spiking/tensor_to_spike_times.h"

#include <vector>
#include <torch/torch.h>


namespace hxtorch::spiking {

/** Transform weight tensor to grenade connections
 */
std::vector<std::vector<std::vector<float>>> tensor_to_spike_times(torch::Tensor times, float dt)
{
	if (times.dim() != 3) {
		throw std::runtime_error("Only data tensors with dim = 3 are supported.");
	}
	if ((times.sizes()[0] == 0) || (times.sizes()[1] == 0) || (times.sizes()[2] == 0)) {
		throw std::runtime_error("Given data tensor has size = 0 along one dimension.");
	}
	if (times.device().is_cuda()) {
		throw std::runtime_error(
		    "The input tensor is expected to be on device torch::device('cpu').");
	}

	std::vector<std::vector<std::vector<float>>> gtimes(
	    times.sizes()[1], std::vector<std::vector<float>>(times.sizes()[2]));
	torch::Tensor const events = torch::nonzero(times);

	auto a_events = events.accessor<long, 2>();
	for (int i = 0; i < events.sizes()[0]; ++i) {
		gtimes.at(a_events[i][1]).at(a_events[i][2]).push_back(dt * a_events[i][0]);
	}

	return gtimes;
}

}
