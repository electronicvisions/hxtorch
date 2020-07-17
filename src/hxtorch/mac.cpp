#include "hxtorch/mac.h"

#include "hxtorch/detail/mac.h"

#include <torch/torch.h>

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/script.h>

namespace hxtorch {

torch::autograd::variable_list MAC::forward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::Variable x,
    torch::autograd::Variable weights,
    int64_t num_sends,
    int64_t wait_between_events)
{
	auto ret = detail::mac_forward(x, weights, num_sends, wait_between_events);
	ctx->save_for_backward({ret, x, weights});
	return {ret};
}

torch::autograd::variable_list MAC::backward(
    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
{
	auto saved_variables = ctx->get_saved_variables();
	auto x = saved_variables[1];
	auto weights = saved_variables[2];

	torch::Tensor gain;
	// scale grad_output with the gain on a per-batch basis
	auto forward_output = saved_variables[0];
	auto torch_output = x.matmul(weights);
	auto elementwise_gain = torch::div(forward_output, torch_output);
	auto mask = torch::logical_or(elementwise_gain > 1, torch::isnan(elementwise_gain));
	if (!mask.all().item().to<bool>()) {
		elementwise_gain.index_put_({mask}, 0);
		gain = elementwise_gain.sum() / torch::logical_not(mask).sum();
		gain.clamp_(0);
	} else {
		// fallback
		gain = torch::tensor(0.);
	}
	return detail::mac_backward(grad_output[0] * gain, x, weights);
}

torch::Tensor mac(
    torch::Tensor const& x,
    torch::Tensor const& weights,
    int64_t const num_sends,
    int64_t wait_between_events)
{
	auto ret = MAC::apply(x, weights, num_sends, wait_between_events);
	return ret[0];
}

static auto registry = torch::RegisterOperators().op("hxtorch::mac", &mac);

} // namespace hxtorch
