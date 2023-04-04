#include "hxtorch/perceptron/mock.h"

#include "hxtorch/perceptron/constants.h"
#include "hxtorch/perceptron/detail/mock.h"
#include "hxtorch/perceptron/matmul.h"
#include <stdexcept>
#include <log4cxx/logger.h>
#include <torch/torch.h>

namespace hxtorch::perceptron {

MockParameter get_mock_parameter()
{
	return detail::getMockParameter();
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
	auto const input_value = constants::input_activation_max / 2.;
	auto const weight_value = constants::synaptic_weight_max / 3.;
	auto const target_output = constants::output_activation_max / 2.;
	// factor by which the input is reduced:
	auto const masking_factor = torch::tensor(0.85);
	int64_t const num_steps = 20;
	int64_t const batch_size = 50;
	auto const rows_per_synram = constants::hardware_matrix_width;
	auto const cols_per_synram = constants::hardware_matrix_height;

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

	auto test_result = matmul(test_input, full_weight, 1, 5);
	// index closest to target output:
	auto best_idx =
	    torch::sub(std::get<0>(torch::median(test_result, 1)), target_output).abs().argmin();

	// compare to vanilla pytorch result in order to calculate the gain
	auto optimal_input = torch::mul(
	    torch::lt(torch::rand({batch_size, rows_per_synram}), torch::pow(masking_factor, best_idx)),
	    input_value);
	auto result_torch = torch::matmul(optimal_input, full_weight);
	auto result_hxtorch = matmul(optimal_input, full_weight, 1, 5);
	auto gain = torch::median(result_hxtorch / result_torch).item<float>();

	auto noise = result_hxtorch - result_torch * gain;
	auto noise_std = torch::median(torch::std(noise, 0, true)).item<float>();

	auto logger = log4cxx::Logger::getLogger("hxtorch.perceptron.measure_mock_parameter");
	LOG4CXX_INFO(
	    logger, "Obtained gain=" << gain << ", noise_std=" << noise_std
	                             << " at output=" << torch::median(result_hxtorch).item<int>());

	return MockParameter(noise_std, gain);
}

} // namespace hxtorch::perceptron
