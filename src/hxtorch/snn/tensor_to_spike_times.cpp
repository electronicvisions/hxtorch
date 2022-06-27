#include "hxtorch/snn/tensor_to_spike_times.h"

#include <vector>
#include <torch/torch.h>


namespace hxtorch::snn {

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

	std::vector<std::vector<std::vector<float>>> gtimes(times.sizes()[0]);

	for (int b = 0; b < times.sizes()[0]; ++b) {
		gtimes.at(b).resize(times.sizes()[2]);
		auto const& batch_element = times.index({b});
		auto batch_times = torch::nonzero(batch_element);

		for (int spike = 0; spike < batch_times.sizes()[0]; ++spike) {
			gtimes.at(b)
			    .at(batch_times.index({spike, 1}).item().toInt())
			    .push_back(batch_times.index({spike, 0}).item().toFloat() * dt);
		}
	}

	return gtimes;
}

}
