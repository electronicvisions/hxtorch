#include <torch/torch.h>

#include "grenade/vx/compute/addition.h"
#include "grenade/vx/config.h"
#include "hxtorch/detail/connection.h"
#include "hxtorch/detail/conversion.h"
#include "hxtorch/detail/inference_tracer.h"

namespace hxtorch::detail {

torch::Tensor add_mock_forward(torch::Tensor const& input, torch::Tensor const& other)
{
	return torch::add(input.floor().clamp(-128., 127.), other.floor().clamp(-128., 127.))
	    .clamp(-128., 127.);
}

namespace {

std::vector<grenade::vx::Int8> convert_add_other(torch::Tensor const& other)
{
	auto const other_1d = other.reshape({-1}).floor().clamp(-128., 127.);
	auto const sizes_1d = other_1d.sizes();

	auto const other_a = other_1d.accessor<float, 1>();
	std::vector<grenade::vx::Int8> other_in(sizes_1d.at(0));
	for (int64_t i = 0; i < sizes_1d.at(0); ++i) {
		other_in[i] = grenade::vx::Int8(other_a[i]);
	}
	return other_in;
}

std::tuple<std::vector<std::vector<grenade::vx::Int8>>, std::vector<int64_t>> convert_add_input(
    torch::Tensor const& input, int64_t const other_size)
{
	detail::tracer_check_input(input);

	auto const sizes = input.sizes().vec();

	auto const input_2d = input.reshape({-1, other_size}).floor().clamp(-128., 127.);
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
torch::Tensor convert_add_output(
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
	ret = ret.reshape(sizes);
	detail::tracer_update_output(ret);
	return ret;
}

}

torch::Tensor add_forward(torch::Tensor const& input, torch::Tensor const& other)
{
	if (other.dim() > input.dim()) {
		throw std::runtime_error("add operation only supports dim(other) <= dim(input).");
	}
	auto inputs = torch::broadcast_tensors({input, other});

	auto const other_in = convert_add_other(inputs.at(1));
	auto const [input_in, sizes_2d] = convert_add_input(inputs.at(0), other_in.size());

	grenade::vx::compute::Addition add(other_in);

	if (!hxtorch::detail::getConnection()) {
		throw std::runtime_error("No connection allocated.");
	}
	auto const results =
	    add.run(input_in, hxtorch::detail::getChip(), *hxtorch::detail::getConnection());
	tracer_add("add", std::move(add));
	return convert_add_output(results, sizes_2d, input.sizes());
}

torch::autograd::variable_list add_backward(
    torch::Tensor const& grad_output,
    torch::Tensor const& /*input*/,
    torch::Tensor const& /*other*/)
{
	return {grad_output, grad_output, {}};
}

} // namespace hxtorch::detail
