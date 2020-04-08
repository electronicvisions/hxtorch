#include "hxtorch/conv.h"

#include "hxtorch/detail/conv.h"
#include "hxtorch/matmul.h"

#include <torch/torch.h>

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
//#include <torch/custom_class.h>
//#include <torch/extension.h>
#include <torch/script.h>

namespace hxtorch {

torch::Tensor conv(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    std::vector<int64_t> stride,
    int64_t const num_sends,
    int64_t const wait_between_events,
    bool mock)
{
	int64_t const dim = stride.size();
	if (weight.dim() != 2 + dim) {
		throw std::runtime_error(
		    "conv expects a weight shape of (out_channels, in_channels, **kernel_shape).");
	}
	if (input.dim() != 2 + dim) {
		throw std::runtime_error(
		    "conv expects a input shape of (minibatches, in_channels, **single_input_shape).");
	}
	if (weight.sizes().vec().at(1) != input.sizes().vec().at(1)) {
		throw std::runtime_error("conv expects matching in_channels in input and weight.");
	}

	auto const weight_sizes = weight.sizes().vec();
	std::vector<int64_t> kernel_size(weight_sizes.begin() + 2, weight_sizes.end());
	auto const input_folded = hxtorch::detail::conv_fold_input(input, kernel_size, stride);
	auto const weight_folded = hxtorch::detail::conv_fold_kernel(weight);

	auto result = matmul(input_folded, weight_folded, num_sends, wait_between_events, mock);
	result = hxtorch::detail::conv_permute_output(result);
	auto input_sizes = input.sizes().vec();
	std::vector<int64_t> input_size(dim);
	std::copy(input_sizes.begin() + 2, input_sizes.end(), input_size.begin());
	result = hxtorch::detail::conv_unfold_output(
	             result, hxtorch::detail::conv_num_outputs(input_size, kernel_size, stride))
	             .contiguous();
	return result;
}

torch::Tensor conv1d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    int64_t stride,
    int64_t num_sends,
    int64_t wait_between_events,
    bool mock)
{
	return conv(input, weight, {stride}, num_sends, wait_between_events, mock);
}

torch::Tensor conv1d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    std::array<int64_t, 1> stride,
    int64_t num_sends,
    int64_t wait_between_events,
    bool mock)
{
	return conv(input, weight, {stride[0]}, num_sends, wait_between_events, mock);
}

torch::Tensor conv2d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    int64_t stride,
    int64_t num_sends,
    int64_t wait_between_events,
    bool mock)
{
	return conv(input, weight, {stride, stride}, num_sends, wait_between_events, mock);
}

torch::Tensor conv2d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    std::array<int64_t, 2> stride,
    int64_t num_sends,
    int64_t wait_between_events,
    bool mock)
{
	return conv(input, weight, {stride[0], stride[1]}, num_sends, wait_between_events, mock);
}

} // namespace hxtorch
