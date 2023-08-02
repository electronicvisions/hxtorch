#include "hxtorch/spiking/types.h"
#include "grenade/vx/common/time.h"
#include "hxtorch/spiking/detail/to_dense.h"
#include <ATen/Functions.h>
#include <ATen/SparseTensorUtils.h>
#include <log4cxx/logger.h>

namespace hxtorch::spiking {

// Specialization for SpikeHandle
torch::Tensor SpikeHandle::to_dense(float runtime, float dt)
{
	return detail::sparse_spike_to_dense(get_data(), batch_size(), population_size(), runtime, dt);
};


// Specialization for CADCHandle
torch::Tensor CADCHandle::to_dense(float runtime, float dt, std::string mode)
{
	// TODO: Use this with overloading or enums/switch?
	if (mode == "linear") {
		// Use linear interpolation
		return detail::sparse_cadc_to_dense_linear(
		    get_data(), batch_size(), population_size(), runtime, dt);
	}
	if (mode == "nn") {
		// Use nearest neighbor interpolation
		return detail::sparse_cadc_to_dense_nn(
		    get_data(), batch_size(), population_size(), runtime, dt);
	}

	std::stringstream ss;
	ss << "Mode '" << mode << "' is not supported yet. Supported modes are: 'linear', 'nn'.";
	throw std::runtime_error(ss.str());
};


// Specialization for CADCHandle returning raw data
std::tuple<torch::Tensor, torch::Tensor> CADCHandle::to_raw()
{
	return detail::sparse_cadc_to_dense_raw(get_data(), batch_size(), population_size());
};


// Dense tensor return if not dt is given
std::tuple<torch::Tensor, float> CADCHandle::to_dense(float runtime, std::string mode)
{
	auto const& data = get_data();

	auto max_time = std::get<1>(*std::max_element(
	    data.begin(), data.end(),
	    [](const std::tuple<int32_t, int64_t, int64_t, int64_t>& a,
	       const std::tuple<int32_t, int64_t, int64_t, int64_t>& b) {
		    return std::get<1>(a) < std::get<2>(b);
	    }));

	float dt = static_cast<float>(max_time) *
	           (static_cast<float>(batch_size() * population_size()) /
	            static_cast<float>(data.size() - batch_size() * population_size())) *
	           1. / 1e6 / static_cast<float>(grenade::vx::common::Time::fpga_clock_cycles_per_us);

	return std::make_tuple(to_dense(runtime, dt, mode), dt);
}


} // namespace hxtorch::spiking
