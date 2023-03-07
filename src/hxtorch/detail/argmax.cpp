#include "hxtorch/detail/argmax.h"

#include "grenade/vx/compute/argmax.h"
#include "hxtorch/detail/connection.h"
#include "hxtorch/detail/conversion.h"
#include "hxtorch/detail/inference_tracer.h"
#include "lola/vx/v3/chip.h"

namespace hxtorch::detail {

torch::Tensor argmax_mock(
    torch::Tensor const& input, c10::optional<int64_t> const dim, bool keepdim)
{
	return torch::argmax(input.floor().clamp(-128., 127.), dim, keepdim);
}

namespace {

std::tuple<std::vector<std::vector<grenade::vx::signal_flow::Int8>>, std::vector<int64_t>>
convert_argmax_input(torch::Tensor const& input, c10::optional<int64_t> const dim)
{
	std::vector<int64_t> dims(input.dim());
	std::iota(dims.begin(), dims.end(), 0);
	std::vector<int64_t> sizes_2d;
	if (dim) {
		// permute such that the dimension for which to calculate argmax is the last
		dims.at(*dim) = dims.size() - 1;
		dims.at(dims.size() - 1) = *dim;

		// reshape such that all but the last dimension are squeezed to 1d
		sizes_2d = {-1, input.sizes().vec().at(*dim)};
	} else {
		// reshape such that all elements reside in the last dimension
		sizes_2d = {1, -1};
	}

	auto const input_2d = input.permute(dims).reshape(sizes_2d).floor().clamp(-128., 127.);
	sizes_2d = input_2d.sizes().vec();

	auto input_a = input_2d.accessor<float, 2>();
	std::vector<std::vector<grenade::vx::signal_flow::Int8>> input_in(sizes_2d.at(0));
	for (int64_t i = 0; i < sizes_2d.at(0); ++i) {
		input_in[i].resize(sizes_2d.at(1));
		for (int64_t j = 0; j < sizes_2d.at(1); ++j) {
			input_in[i][j] = grenade::vx::signal_flow::Int8(input_a[i][j]);
		}
	}
	return {input_in, sizes_2d};
}

torch::Tensor convert_argmax_output(
    std::vector<std::vector<grenade::vx::signal_flow::UInt32>> const& results,
    c10::IntArrayRef const& sizes_2d,
    c10::IntArrayRef const& sizes,
    c10::optional<int64_t> const dim,
    bool const keepdim)
{
	torch::Tensor ret;
	if (dim) {
		ret = torch::zeros({sizes_2d.at(0), 1}, torch::kLong);
		auto result_a = ret.accessor<int64_t, 2>();
		for (int64_t i = 0; i < sizes_2d.at(0); ++i) {
			result_a[i][0] = static_cast<int64_t>(results[i][0]);
		}
		std::vector<int64_t> dims(sizes.size());
		std::iota(dims.begin(), dims.end(), 0);
		dims.at(*dim) = dims.size() - 1;
		dims.at(dims.size() - 1) = *dim;
		auto ret_sizes = sizes.vec();
		ret_sizes.at(*dim) = ret_sizes.at(ret_sizes.size() - 1);
		ret_sizes.at(ret_sizes.size() - 1) = 1;
		ret = ret.reshape(ret_sizes).permute(dims);
		if (!keepdim) {
			ret = ret.squeeze(*dim);
		}
	} else {
		ret = torch::tensor(static_cast<int64_t>(results.at(0).at(0)), torch::kLong);
	}
	return ret;
}

} // namespace

torch::Tensor argmax(torch::Tensor const& input, c10::optional<int64_t> const dim, bool keepdim)
{
	if (has_tracer() && !(input.dim() == 2 && dim && *dim == 1 && keepdim)) {
		throw std::runtime_error(
		    "Tracing argmax operation only supported for 2d input with dim=1 and keepdim enabled.");
	}

	tracer_check_input(input);
	auto const [input_in, sizes_2d] = convert_argmax_input(input, dim);

	grenade::vx::compute::ArgMax kernel(sizes_2d.at(1));
	if (!hxtorch::detail::getExecutor()) {
		throw std::runtime_error("No connection allocated.");
	}
	auto const results =
	    kernel.run(input_in, hxtorch::detail::getChip(), *hxtorch::detail::getExecutor());
	tracer_add("argmax", std::move(kernel));
	auto const ret = convert_argmax_output(results, sizes_2d, input.sizes(), dim, keepdim);
	tracer_update_output(ret);
	return ret;
}

} // namespace hxtorch::detail
