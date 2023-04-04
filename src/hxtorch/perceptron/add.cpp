#include "hxtorch/perceptron/add.h"

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "hxtorch/perceptron/detail/add.h"

namespace hxtorch::perceptron {

class Add : public torch::autograd::Function<Add>
{
public:
	static torch::autograd::variable_list forward(
	    torch::autograd::AutogradContext* ctx,
	    torch::autograd::Variable input,
	    torch::autograd::Variable other,
	    bool mock)
	{
		ctx->save_for_backward({input, other});
		return {mock ? detail::add_mock_forward(input, other) : detail::add_forward(input, other)};
	}

	static torch::autograd::variable_list backward(
	    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
	{
		auto saved_variables = ctx->get_saved_variables();
		auto input = saved_variables[0];
		auto other = saved_variables[0];
		return detail::add_backward(grad_output[0], input, other);
	}
};


torch::Tensor add(
    torch::Tensor const& input, torch::Tensor const& other, double const alpha, bool const mock)
{
	if (alpha != 1.) {
		throw std::runtime_error("add only supports alpha = 1.");
	}
	auto ret = Add::apply(input, other, mock);
	return ret[0];
}

static auto registry = torch::RegisterOperators().op("hxtorch_perceptron::add", &add);


} // namespace hxtorch::perceptron
