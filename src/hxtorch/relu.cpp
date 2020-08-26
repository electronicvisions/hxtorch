#include "hxtorch/relu.h"

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "hxtorch/detail/relu.h"

namespace hxtorch {

class ReLU : public torch::autograd::Function<ReLU>
{
public:
	static torch::autograd::variable_list forward(
	    torch::autograd::AutogradContext* ctx, torch::autograd::Variable input, bool mock)
	{
		ctx->save_for_backward({input});
		return {mock ? detail::relu_mock_forward(input) : detail::relu_forward(input)};
	}

	static torch::autograd::variable_list backward(
	    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
	{
		auto saved_variables = ctx->get_saved_variables();
		auto input = saved_variables[0];
		return detail::relu_backward(grad_output[0], input);
	}
};


torch::Tensor relu(torch::Tensor const& input, bool const mock)
{
	auto ret = ReLU::apply(input, mock);
	return ret[0];
}

static auto registry = torch::RegisterOperators().op("hxtorch::relu", &relu);

} // namespace hxtorch
