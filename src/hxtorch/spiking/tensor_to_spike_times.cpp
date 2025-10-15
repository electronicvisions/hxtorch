#include "hxtorch/spiking/tensor_to_spike_times.h"
#include "grenade/vx/common/time.h"

#include <vector>
#include <torch/torch.h>


namespace hxtorch::spiking {

/** Transform weight tensor to grenade connections
 */
std::vector<std::vector<std::vector<grenade::vx::common::Time>>> tensor_to_spike_times(
    torch::Tensor times, float dt)
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

	std::vector<std::vector<std::vector<grenade::vx::common::Time>>> gtimes(
	    times.sizes()[1], std::vector<std::vector<grenade::vx::common::Time>>(times.sizes()[2]));
	torch::Tensor const events = torch::nonzero(times);

	auto a_events = events.accessor<long, 2>();
	for (int i = 0; i < events.sizes()[0]; ++i) {
		uint64_t time_value = grenade::vx::common::Time::fpga_clock_cycles_per_us *
		                      static_cast<uint64_t>(dt * a_events[i][0] * 1e6);
		gtimes.at(a_events[i][1])
		    .at(a_events[i][2])
		    .push_back(grenade::vx::common::Time(time_value));
	}

	return gtimes;
}

}
