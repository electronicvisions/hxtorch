#include "hxtorch/detail/mac.h"

#include "grenade/vx/compute_single_mac.h"
#include "grenade/vx/config.h"
#include "hxtorch/detail/connection.h"
#include "hxtorch/detail/conversion.h"

namespace hxtorch::detail {

torch::Tensor mac_forward(
    torch::Tensor x, torch::Tensor weights, int64_t num_sends, int64_t wait_between_events)
{
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

	std::vector<std::vector<haldls::vx::SynapseQuad::Weight>> m_weights{
	    num_rows, std::vector<haldls::vx::SynapseQuad::Weight>{num_cols}};

	// TODO: let's assume it's floats...
	auto weights_a = weights.accessor<float, 2>();
	for (size_t i = 0; i < num_double_rows; i++) {
		for (size_t j = 0; j < num_cols; j++) {
			auto const signed_weight = convert_weight(weights_a[i][j]);
			m_weights[2 * i][j] = signed_weight.positive;
			m_weights[2 * i + 1][j] = signed_weight.negative;
		}
	}

	std::vector<haldls::vx::SynapseDriverConfig::RowMode> row_modes(num_rows);
	for (size_t i = 0; i < num_double_rows; i++) {
		row_modes[2 * i] = haldls::vx::SynapseDriverConfig::RowMode::excitatory;
		row_modes[2 * i + 1] = haldls::vx::SynapseDriverConfig::RowMode::inhibitory;
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

	grenade::vx::ComputeSingleMAC mac{m_weights, row_modes, hxtorch::detail::getChip(),
	                                  static_cast<size_t>(num_sends),
	                                  haldls::vx::Timer::Value(wait_between_events)};

	if (!hxtorch::detail::getConnection()) {
		throw std::runtime_error("No connection allocated.");
	}
	auto ret = torch::zeros({static_cast<int64_t>(num_inputs), static_cast<int64_t>(num_cols)});
	auto ret_a = ret.accessor<float, 2>();
	auto const results = mac.run(xin, *hxtorch::detail::getConnection());
	for (size_t input = 0; input < num_inputs; ++input) {
		for (size_t i = 0; i < results.at(0).size(); i++) {
			ret_a[input][i] = convert_membrane(results[input][i]);
		}
	}
	if (x_initial_dim == 1) {
		ret.squeeze_(0);
	}
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
	return {grad_x, grad_weights, {}, {}};
}

} // namespace hxtorch::detail
