#include "hxtorch/perceptron/mac.h"

#include "hxtorch/perceptron/detail/mac.h"
#include "hxtorch/perceptron/detail/mock.h"

#include <torch/torch.h>

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/script.h>

namespace hxtorch::perceptron {

torch::autograd::variable_list MAC::forward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::Variable x,
    torch::autograd::Variable weights,
    int64_t num_sends,
    int64_t wait_between_events,
    bool mock,
    int64_t madc_recording_neuron_id,
    std::string madc_recording_path)
{
	if (mock && (madc_recording_path != "")) {
		throw std::runtime_error("Recording with MADC in mock-mode is not implemented.");
	}
	auto ret = mock ? detail::mac_mock_forward(x, weights, num_sends)
	                : detail::mac_forward(
	                      x, weights, num_sends, wait_between_events, madc_recording_neuron_id,
	                      madc_recording_path);
	ctx->save_for_backward({x, weights, torch::tensor(num_sends)});
	return {ret};
}

torch::autograd::variable_list MAC::backward(
    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
{
	auto saved_variables = ctx->get_saved_variables();
	auto x = saved_variables[0];
	auto weights = saved_variables[1];
	auto num_sends = saved_variables[2];
	auto gain = detail::getMockParameter().gain * num_sends;
	return detail::mac_backward(grad_output[0] * gain, x, weights);
}

torch::Tensor mac(
    torch::Tensor const& x,
    torch::Tensor const& weights,
    int64_t const num_sends,
    int64_t const wait_between_events,
    bool const mock,
    int64_t madc_recording_neuron_id,
    std::string madc_recording_path)
{
	auto ret = MAC::apply(
	    x, weights, num_sends, wait_between_events, mock, madc_recording_neuron_id,
	    madc_recording_path);
	return ret[0];
}

static auto registry = torch::RegisterOperators().op("hxtorch_perceptron::mac", &mac);

} // namespace hxtorch::perceptron
