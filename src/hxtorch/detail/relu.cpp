#include <torch/torch.h>

#include "grenade/vx/compute_single_converting_relu.h"
#include "grenade/vx/compute_single_relu.h"
#include "hxtorch/detail/connection.h"
#include "hxtorch/detail/conversion.h"

namespace hxtorch::detail {

torch::Tensor relu_mock_forward(torch::Tensor const& input)
{
	return input.floor().clamp(0., 127.);
}

namespace {

std::tuple<std::vector<std::vector<grenade::vx::Int8>>, std::vector<int64_t>> convert_relu_input(
    torch::Tensor const& input)
{
	auto const sizes = input.sizes();

	auto const input_2d =
	    input.reshape({-1, sizes.at(sizes.size() - 1)}).floor().clamp(-128., 127.);
	auto const sizes_2d = input_2d.sizes();

	auto const input_a = input_2d.accessor<float, 2>();
	std::vector<std::vector<grenade::vx::Int8>> input_in(sizes_2d.at(0));
	for (int64_t i = 0; i < sizes_2d.at(0); ++i) {
		input_in[i].resize(sizes_2d.at(1));
		for (int64_t j = 0; j < sizes_2d.at(1); ++j) {
			input_in[i][j] = grenade::vx::Int8(input_a[i][j]);
		}
	}
	return {input_in, sizes_2d.vec()};
}

template <typename T>
torch::Tensor convert_relu_output(
    std::vector<std::vector<T>> const& results,
    std::vector<int64_t> const& sizes_2d,
    c10::IntArrayRef const& sizes)
{
	auto ret = torch::zeros(sizes_2d);
	auto ret_a = ret.accessor<float, 2>();
	for (int64_t i = 0; i < sizes_2d.at(0); ++i) {
		for (int64_t j = 0; j < sizes_2d.at(1); ++j) {
			ret_a[i][j] = static_cast<float>(results[i][j]);
		}
	}
	return ret.reshape(sizes);
}

}

torch::Tensor relu_forward(torch::Tensor const& input)
{
	auto const [input_in, sizes_2d] = convert_relu_input(input);

	grenade::vx::ComputeSingleReLU relu(sizes_2d.at(1));

	if (!hxtorch::detail::getConnection()) {
		throw std::runtime_error("No connection allocated.");
	}
	auto const results = relu.run(input_in, *hxtorch::detail::getConnection());

	return convert_relu_output(results, sizes_2d, input.sizes());
}

torch::autograd::variable_list relu_backward(
    torch::Tensor const& grad_output, torch::Tensor const& input)
{
	auto const gt = torch::gt(input, 0);
	return {grad_output * gt, {}};
}


torch::Tensor converting_relu_mock_forward(torch::Tensor const& input, int64_t const shift)
{
	return torch::div(input.floor().clamp(0., 127.), std::pow(2, shift)).floor().clamp(0., 31.);
}

torch::Tensor converting_relu_forward(torch::Tensor const& input, int64_t const shift)
{
	if (shift < 0) {
		throw std::runtime_error("Non-negative shift value expected.");
	}

	auto const [input_in, sizes_2d] = convert_relu_input(input);

	grenade::vx::ComputeSingleConvertingReLU converting_relu(sizes_2d.at(1), shift);

	if (!hxtorch::detail::getConnection()) {
		throw std::runtime_error("No connection allocated.");
	}
	auto const results = converting_relu.run(input_in, *hxtorch::detail::getConnection());

	return convert_relu_output(results, sizes_2d, input.sizes());
}

torch::autograd::variable_list converting_relu_backward(
    torch::Tensor const& grad_output, torch::Tensor const& input, int64_t const shift)
{
	auto const gt = torch::gt(input, 0);
	return {torch::div(grad_output * gt, std::pow(2, shift)), {}, {}};
}

} // namespace hxtorch::detail
