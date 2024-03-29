#include "hxtorch/perceptron/detail/mac.h"

#include "grenade/vx/compute/mac.h"
#include "grenade/vx/signal_flow/event.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "hxtorch/core/detail/connection.h"
#include "hxtorch/perceptron/constants.h"
#include "hxtorch/perceptron/detail/conversion.h"
#include "hxtorch/perceptron/detail/inference_tracer.h"
#include "hxtorch/perceptron/detail/mock.h"
#include "lola/vx/v3/chip.h"

#include "hate/timer.h"

namespace hxtorch::perceptron::detail {

torch::Tensor mac_mock_forward(
    torch::Tensor const& x, torch::Tensor const& weights, int64_t num_sends)
{
	if (weights.dim() != 2) {
		throw std::runtime_error("HICperceptron-X only supports 2D weight matrices");
	}
	if (x.dim() != 1 && x.dim() != 2) {
		throw std::runtime_error("HICperceptron-X only supports 1D or 2D input");
	}

	// quantize weights and inputs
	auto const quantized_weights = weights.round();
	auto const quantized_inputs = x.round();

	if ((quantized_weights.min().item().to<float>() <
	     hxtorch::perceptron::constants::synaptic_weight_min) ||
	    (quantized_weights.max().item().to<float>() >
	     hxtorch::perceptron::constants::synaptic_weight_max)) {
		throw std::overflow_error(
		    "HICperceptron-X only supports weights between " +
		    std::to_string(hxtorch::perceptron::constants::synaptic_weight_min) + " and " +
		    std::to_string(hxtorch::perceptron::constants::synaptic_weight_max) + ", got " +
		    std::to_string(quantized_weights.min().item().to<float>()) + " (min), " +
		    std::to_string(quantized_weights.max().item().to<float>()) + " (max)");
	}
	if ((quantized_inputs.min().item().to<float>() <
	     static_cast<int>(hxtorch::perceptron::constants::input_activation_min)) ||
	    (quantized_inputs.max().item().to<float>() >
	     static_cast<int>(hxtorch::perceptron::constants::input_activation_max))) {
		throw std::overflow_error(
		    "HICperceptron-X only supports inputs between " +
		    std::to_string(constants::input_activation_min) + " and " +
		    std::to_string(constants::input_activation_max) + ", got " +
		    std::to_string(quantized_inputs.min().item().to<float>()) + " (min), " +
		    std::to_string(quantized_inputs.max().item().to<float>()) + " (max)");
	}

	// split one synram height from matrix
	auto const split_inputs = quantized_inputs.split(
	    hxtorch::perceptron::constants::hardware_matrix_height, quantized_inputs.dim() - 1);
	auto const split_weights =
	    quantized_weights.split(hxtorch::perceptron::constants::hardware_matrix_height, 0);

	size_t const num_cols = quantized_weights.sizes().vec().at(1);
	auto output_sizes = quantized_inputs.sizes().vec();
	output_sizes.back() = num_cols;
	torch::Tensor results = torch::zeros(output_sizes, torch::TensorOptions().device(x.device()));
	for (size_t i = 0; i < split_inputs.size(); ++i) {
		// perform matrix multiplication for one synram vertical split
		auto local_results = split_inputs.at(i).matmul(split_weights.at(i));
		// multiply with constant analog multiplication gain
		local_results.mul_(static_cast<float>(num_sends) * getMockParameter().gain);
		// add membrane noise
		if (getMockParameter().noise_std > 0.) {
			auto const noise = torch::normal(
			    0., getMockParameter().noise_std, results.sizes(), c10::nullopt,
			    torch::TensorOptions().device(x.device()));
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
    torch::Tensor x,
    torch::Tensor weights,
    int64_t num_sends,
    int64_t wait_between_events,
    int64_t madc_recording_neuron_id,
    std::string madc_recording_path)
{
	detail::tracer_check_input(x);

	if (weights.dim() != 2) {
		throw std::runtime_error("HICperceptron-X only supports 2D weight matrices");
	}

	size_t const x_initial_dim = x.dim();
	if (x.dim() == 1) {
		x = x.unsqueeze(0);
	} else if (x.dim() != 2) {
		throw std::runtime_error("HICperceptron-X only supports 1D or 2D input");
	}

	// create vector
	size_t const num_rows = weights.sizes().vec().at(0);
	size_t const num_cols = weights.sizes().vec().at(1);

	if (static_cast<size_t>(x.sizes().vec().back()) != num_rows) {
		throw std::runtime_error("HICperceptron-X only supports input lengths that match the "
		                         "corresponding weight matrix dim size");
	}

	grenade::vx::compute::MAC::Weights m_weights{
	    num_rows, grenade::vx::compute::MAC::Weights::value_type{num_cols}};

	// TODO: let's assume it's floats...
	auto weights_a = weights.accessor<float, 2>();
	for (size_t i = 0; i < num_rows; i++) {
		for (size_t j = 0; j < num_cols; j++) {
			m_weights[i][j] = convert_weight(weights_a[i][j]);
		}
	}

	size_t const num_inputs = x.sizes().vec().at(0);
	std::vector<std::vector<grenade::vx::signal_flow::UInt5>> xin(num_inputs);
	for (size_t input = 0; input < num_inputs; ++input) {
		xin[input].resize(num_rows);
	}

	// TODO: let's assume it's floats...
	auto x_a = x.accessor<float, 2>();
	for (size_t input = 0; input < num_inputs; ++input) {
		for (size_t i = 0; i < num_rows; i++) {
			auto const activation = convert_activation(x_a[input][i]);
			xin[input][i] = activation;
		}
	}

	grenade::vx::compute::MAC mac{
	    std::move(m_weights),
	    static_cast<size_t>(num_sends),
	    grenade::vx::common::Time(wait_between_events),
	    false,
	    halco::hicann_dls::vx::v3::AtomicNeuronOnDLS(halco::common::Enum(madc_recording_neuron_id)),
	    madc_recording_path};

	if (!hxtorch::core::detail::getExecutor()) {
		throw std::runtime_error("No connection allocated.");
	}
	auto ret = torch::zeros({static_cast<int64_t>(num_inputs), static_cast<int64_t>(num_cols)});
	auto ret_a = ret.accessor<float, 2>();
	auto const results =
	    mac.run(xin, hxtorch::core::detail::getChip(), *hxtorch::core::detail::getExecutor());
	tracer_add("mac", std::move(mac));
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
	return {grad_x, grad_weights, {}, {}, {}, {}, {}};
}

} // namespace hxtorch::perceptron::detail
