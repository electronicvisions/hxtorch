#include "hxtorch/detail/mac.h"

#include "grenade/vx/compute_single_mac.h"
#include "grenade/vx/config.h"
#include "grenade/vx/event.h"
#include "hxtorch/detail/connection.h"
#include "hxtorch/detail/conversion.h"
#include "hxtorch/detail/inference_tracer.h"
#include "hxtorch/detail/mock.h"

#include "hate/timer.h"

namespace hxtorch::detail {

torch::Tensor mac_mock_forward(
    torch::Tensor const& x, torch::Tensor const& weights, int64_t num_sends)
{
	if (weights.dim() != 2) {
		throw std::runtime_error("HICANN-X only supports 2D weight matrices");
	}
	if (x.dim() != 1 && x.dim() != 2) {
		throw std::runtime_error("HICANN-X only supports 1D or 2D input");
	}

	// quantize weights and inputs
	auto const quantized_weights = weights.round();
	auto const quantized_inputs = x.round();

	if ((quantized_weights.min().item().to<float>() < -haldls::vx::v2::SynapseQuad::Weight::max) ||
	    (quantized_weights.max().item().to<float>() > haldls::vx::v2::SynapseQuad::Weight::max)) {
		throw std::overflow_error(
		    "HICANN-X only supports weights between " +
		    std::to_string(-haldls::vx::v2::SynapseQuad::Weight::max) + " and " +
		    std::to_string(haldls::vx::v2::SynapseQuad::Weight::max) + ", got " +
		    std::to_string(quantized_weights.min().item().to<float>()) + " (min), " +
		    std::to_string(quantized_weights.max().item().to<float>()) + " (max)");
	}
	if ((quantized_inputs.min().item().to<float>() < static_cast<int>(grenade::vx::UInt5::min)) ||
	    (quantized_inputs.max().item().to<float>() > static_cast<int>(grenade::vx::UInt5::max))) {
		throw std::overflow_error(
		    "HICANN-X only supports inputs between " + std::to_string(grenade::vx::UInt5::min) +
		    " and " + std::to_string(grenade::vx::UInt5::max) + ", got " +
		    std::to_string(quantized_inputs.min().item().to<float>()) + " (min), " +
		    std::to_string(quantized_inputs.max().item().to<float>()) + " (max)");
	}

	// split one synram height from matrix
	auto const rows_per_synram = halco::hicann_dls::vx::SynapseRowOnSynram::size /
	                             halco::hicann_dls::vx::SynapseRowOnSynapseDriver::size;
	auto const split_inputs = quantized_inputs.split(rows_per_synram, quantized_inputs.dim() - 1);
	auto const split_weights = quantized_weights.split(rows_per_synram, 0);

	size_t const num_cols = quantized_weights.sizes().vec().at(1);
	auto output_sizes = quantized_inputs.sizes().vec();
	output_sizes.back() = num_cols;
	torch::Tensor results = torch::zeros(output_sizes);
	for (size_t i = 0; i < split_inputs.size(); ++i) {
		// perform matrix multiplication for one synram vertical split
		auto local_results = split_inputs.at(i).matmul(split_weights.at(i));
		// multiply with constant analog multiplication gain
		local_results.mul_(static_cast<float>(num_sends) * getMockParameter().gain);
		// add membrane noise
		if (getMockParameter().noise_std > 0.) {
			auto const noise = torch::normal(0., getMockParameter().noise_std, results.sizes());
			local_results.add_(noise);
		}
		// digitize membrane potential
		local_results.floor_().clamp_(-128., 127.);
		// perform digital addition of synram vertical split
		results.add_(local_results);
	}
	// restrict result to int8 range
	results.clamp_(-128., 127.);
	return results;
}

torch::Tensor mac_forward(
    torch::Tensor x, torch::Tensor weights, int64_t num_sends, int64_t wait_between_events)
{
	detail::tracer_check_input(x);

	if (weights.dim() != 2) {
		throw std::runtime_error("HICANN-X only supports 2D weight matrices");
	}

	size_t const x_initial_dim = x.dim();
	if (x.dim() == 1) {
		x = x.unsqueeze(0);
	} else if (x.dim() != 2) {
		throw std::runtime_error("HICANN-X only supports 1D or 2D input");
	}

	// create vector
	size_t const num_double_rows = weights.sizes().vec().at(0);
	size_t const num_rows = 2 * num_double_rows;
	size_t const num_cols = weights.sizes().vec().at(1);

	if (static_cast<size_t>(x.sizes().vec().back()) != num_double_rows) {
		throw std::runtime_error("HICANN-X only supports input lengths that match the "
		                         "corresponding weight matrix dim size");
	}

	grenade::vx::ComputeSingleMAC::Weights m_weights{
	    num_rows, grenade::vx::ComputeSingleMAC::Weights::value_type{num_cols}};

	// TODO: let's assume it's floats...
	auto weights_a = weights.accessor<float, 2>();
	for (size_t i = 0; i < num_double_rows; i++) {
		for (size_t j = 0; j < num_cols; j++) {
			auto const signed_weight = convert_weight(weights_a[i][j]);
			m_weights[2 * i][j] = signed_weight.positive;
			m_weights[2 * i + 1][j] = signed_weight.negative;
		}
	}

	grenade::vx::ComputeSingleMAC::RowModes row_modes(num_rows);
	for (size_t i = 0; i < num_double_rows; i++) {
		row_modes[2 * i] = grenade::vx::ComputeSingleMAC::RowModes::value_type::excitatory;
		row_modes[2 * i + 1] = grenade::vx::ComputeSingleMAC::RowModes::value_type::inhibitory;
	}

	size_t const num_inputs = x.sizes().vec().at(0);
	std::vector<std::vector<grenade::vx::UInt5>> xin(num_inputs);
	for (size_t input = 0; input < num_inputs; ++input) {
		xin[input].resize(num_rows);
	}

	// TODO: let's assume it's floats...
	auto x_a = x.accessor<float, 2>();
	for (size_t input = 0; input < num_inputs; ++input) {
		for (size_t i = 0; i < num_double_rows; i++) {
			auto const activation = convert_activation(x_a[input][i]);
			xin[input][2 * i] = activation;
			xin[input][2 * i + 1] = activation;
		}
	}

	// only add name of operation
	for (auto& tracer : detail::getInferenceTracer()) {
		assert(tracer);
		tracer->operation_names.push_back("mac");
	}

	grenade::vx::ComputeSingleMAC mac{m_weights, row_modes, static_cast<size_t>(num_sends),
	                                  grenade::vx::TimedSpike::Time(wait_between_events)};

	if (!hxtorch::detail::getConnection()) {
		throw std::runtime_error("No connection allocated.");
	}
	auto ret = torch::zeros({static_cast<int64_t>(num_inputs), static_cast<int64_t>(num_cols)});
	auto ret_a = ret.accessor<float, 2>();
	auto const results =
	    mac.run(xin, hxtorch::detail::getChip(), *hxtorch::detail::getConnection());
	for (size_t input = 0; input < num_inputs; ++input) {
		for (size_t i = 0; i < results.at(0).size(); i++) {
			ret_a[input][i] = convert_membrane(results[input][i]);
		}
	}
	if (x_initial_dim == 1) {
		ret.squeeze_(0);
	}
	detail::tracer_update_output(ret);
	return ret;
}

torch::autograd::variable_list mac_backward(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor weights)
{
	auto grad_x = grad_output.matmul(weights.t());
	if (x.dim() == 1) {
		x = x.unsqueeze(0);
	}
	if (grad_output.dim() == 1) {
		grad_output = grad_output.unsqueeze(0);
	}
	auto grad_weights = x.t().matmul(grad_output);
	return {grad_x, grad_weights, {}, {}, {}};
}

} // namespace hxtorch::detail
