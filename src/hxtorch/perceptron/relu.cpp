#include "hxtorch/perceptron/relu.h"

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "hxtorch/perceptron/detail/relu.h"

namespace hxtorch::perceptron {

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


class ConvertingReLU : public torch::autograd::Function<ConvertingReLU>
{
public:
	static torch::autograd::variable_list forward(
	    torch::autograd::AutogradContext* ctx,
	    torch::autograd::Variable input,
	    int64_t shift,
	    bool mock)
	{
		ctx->save_for_backward({input, torch::tensor(shift)});
		return {
		    mock ? detail::converting_relu_mock_forward(input, shift)
		         : detail::converting_relu_forward(input, shift)};
	}

	static torch::autograd::variable_list backward(
	    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
	{
		auto saved_variables = ctx->get_saved_variables();
		auto input = saved_variables[0];
		auto shift = saved_variables[1];
		return detail::converting_relu_backward(grad_output[0], input, shift.item().to<int64_t>());
	}
};


torch::Tensor converting_relu(torch::Tensor const& input, int64_t const shift, bool const mock)
{
	auto ret = ConvertingReLU::apply(input, shift, mock);
	return ret[0];
}

static auto registry = torch::RegisterOperators()
                           .op("hxtorch_perceptron::relu", &relu)
                           .op("hxtorch_perceptron::converting_relu", &converting_relu);

} // namespace hxtorch::perceptron
