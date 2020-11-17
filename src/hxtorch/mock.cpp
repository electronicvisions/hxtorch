#include "hxtorch/mock.h"

#include "grenade/vx/types.h"
#include "halco/hicann-dls/vx/v2/synapse.h"
#include "haldls/vx/v2/synapse.h"
#include "hxtorch/detail/mock.h"
#include "hxtorch/matmul.h"
#include <stdexcept>
#include <log4cxx/logger.h>
#include <torch/torch.h>

namespace hxtorch {

MockParameter get_mock_parameter()
{
	return hxtorch::detail::getMockParameter();
}

void set_mock_parameter(MockParameter const& parameter)
{
	if ((parameter.gain <= 0) || (parameter.gain > 1)) {
		throw std::overflow_error(
		    "Gain is expected to be in the interval (0, 1] but was " +
		    std::to_string(parameter.gain));
	}
	detail::getMockParameter() = parameter;
}

MockParameter measure_mock_parameter()
{
	// parameters for measurement
	auto const input_value = grenade::vx::UInt5::max / 2.;
	auto const weight_value = haldls::vx::v2::SynapseQuad::Weight::max / 3.;
	auto const target_output = grenade::vx::Int8::max / 2.;
	// factor by which the input is reduced:
	auto const masking_factor = torch::tensor(0.85);
	int64_t const num_steps = 20;
	int64_t const batch_size = 50;
	auto const rows_per_synram = halco::hicann_dls::vx::v2::SynapseRowOnSynram::size / 2;
	auto const cols_per_synram = halco::hicann_dls::vx::v2::SynapseOnSynapseRow::size;

	// full weight matrix:
	auto full_weight =
	    torch::full({rows_per_synram, cols_per_synram}, weight_value, torch::kFloat32);
	// vector with decreasing mean value to determine optimal input:
	auto test_input = torch::empty({num_steps, rows_per_synram}, torch::kFloat32);
	for (int64_t i = 0; i < num_steps; ++i) {
		test_input.index_put_(
		    {i}, torch::mul(
		             torch::lt(torch::rand({rows_per_synram}), torch::pow(masking_factor, i)),
		             input_value));
	}

	auto test_result = hxtorch::matmul(test_input, full_weight, 1, 5);
	// index closest to target output:
	auto best_idx =
	    torch::sub(std::get<0>(torch::median(test_result, 1)), target_output).abs().argmin();

	// compare to vanilla pytorch result in order to calculate the gain
	auto optimal_input = torch::mul(
	    torch::lt(torch::rand({batch_size, rows_per_synram}), torch::pow(masking_factor, best_idx)),
	    input_value);
	auto result_torch = torch::matmul(optimal_input, full_weight);
	auto result_hxtorch = hxtorch::matmul(optimal_input, full_weight, 1, 5);
	auto gain = torch::median(result_hxtorch / result_torch).item<float>();

	auto noise = result_hxtorch - result_torch * gain;
	auto noise_std = torch::median(torch::std(noise, 0, true)).item<float>();

	auto logger = log4cxx::Logger::getLogger("hxtorch.measure_mock_parameter");
	LOG4CXX_INFO(
	    logger, "Obtained gain=" << gain << ", noise_std=" << noise_std
	                             << " at output=" << torch::median(result_hxtorch).item<int>());

	return MockParameter(noise_std, gain);
}

} // namespace hxtorch
