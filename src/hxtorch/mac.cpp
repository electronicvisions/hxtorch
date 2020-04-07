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
	ctx->save_for_backward({x, weights});
	auto ret = detail::mac_forward(x, weights, num_sends, wait_between_events);
	return {ret};
}

torch::autograd::variable_list MAC::backward(
    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
{
	auto saved_variables = ctx->get_saved_variables();
	auto x = saved_variables[0];
	auto weights = saved_variables[1];
	return detail::mac_backward(grad_output[0], x, weights);
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
