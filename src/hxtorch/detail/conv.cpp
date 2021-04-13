#include "hxtorch/detail/conv.h"

#include "hxtorch/detail/conv1d.h"
#include "hxtorch/detail/iterator.h"
#include "hxtorch/detail/mac.h"
#include "hxtorch/detail/narrow.h"
#include "hxtorch/matmul.h"

#include <torch/torch.h>

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/script.h>

namespace hxtorch::detail {

std::vector<int64_t> conv_output_size(
    std::vector<int64_t> input_size,
    std::vector<int64_t> kernel_size,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation)
{
	auto const dim{input_size.size()};
	if (kernel_size.size() != dim || stride.size() != dim || dilation.size() != dim) {
		throw std::runtime_error("conv_output_size: sizes of arguments don't match.");
	}
	std::vector<int64_t> output_size(dim);
	for (size_t i = 0; i < dim; ++i) {
		output_size.at(i) =
		    conv1d_output_size(input_size.at(i), kernel_size.at(i), stride.at(i), dilation.at(i));
	}
	return output_size;
}

torch::Tensor conv(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    std::vector<int64_t> const& stride,
    std::vector<int64_t> const& dilation,
    int64_t const num_sends,
    int64_t const wait_between_events,
    bool const mock)
{
	int64_t const groups(1); // preparation for groups
	int64_t const dim = stride.size();
	if (weight.dim() != 2 + dim) {
		throw std::runtime_error(
		    "conv expects a weight shape of (out_channels, in_channels, **kernel_shape).");
	}
	if (input.dim() != 2 + dim) {
		throw std::runtime_error(
		    "conv expects a input shape of (minibatches, in_channels, **single_input_shape).");
	}
	if (groups < 1) {
		throw std::runtime_error("conv expects groups >= 1.");
	}
	// get dimensions from input
	int64_t const batch_size{input.sizes().at(0)};
	int64_t const in_channels{input.sizes().at(1)};
	std::vector<int64_t> const input_size(input.sizes().begin() + 2, input.sizes().end());
	// get dimensions from weight kernel
	int64_t const out_channels{weight.sizes().at(0)};
	int64_t const out_channels_group{out_channels / groups}; // per group
	int64_t const in_channels_group{weight.sizes().at(1)};   // per group
	std::vector<int64_t> const kernel_size(weight.sizes().begin() + 2, weight.sizes().end());

	if (in_channels % groups != 0) {
		throw std::runtime_error(
		    "Input channels: " + std::to_string(in_channels) +
		    ". Groups: " + std::to_string(groups) +
		    ". The number of input channels is not divisible by number of groups");
	}
	if (out_channels % groups != 0) {
		throw std::runtime_error(
		    "Output channels: " + std::to_string(out_channels) +
		    ". Groups: " + std::to_string(groups) +
		    ". The number of output channels is not divisible by number of groups");
	}
	if (in_channels != in_channels_group * groups) {
		throw std::runtime_error("conv expects matching in_channels in input and weight.");
	}

	// this also checks for correct dimensions of its arguments
	std::vector<int64_t> const output_size{
	    hxtorch::detail::conv_output_size(input_size, kernel_size, stride, dilation)};

	for (int64_t i{0}; i < dim; ++i) {
		if (kernel_size.at(i) > input_size.at(i)) {
			throw std::runtime_error(
			    "Input size per channel: " + std::to_string(input_size.at(i)) +
			    ". Kernel size: " + std::to_string(kernel_size.at(i)) + ". " +
			    "Kernel size can't be greater than actual input size");
		}
	}

	// get strides from input
	int64_t const batch_stride{input.strides().at(0)};
	int64_t const in_channels_stride{input.strides().at(1)};
	std::vector<int64_t> const input_stride(input.strides().begin() + 2, input.strides().end());

	// calculate new sizes for strided view on the input
	std::vector<int64_t> input_view_sizes(2 + 2 * dim);
	input_view_sizes.at(0) = batch_size;
	input_view_sizes.at(dim + 1) = in_channels_group;
	for (int64_t i{0}; i < dim; ++i) {
		input_view_sizes.at(i + 1) = output_size.at(i);
		input_view_sizes.at(i + dim + 2) = kernel_size.at(i);
	}

	// calculate new in-memory strides for strided view on the input
	std::vector<int64_t> input_view_strides(input_view_sizes.size());
	input_view_strides.at(0) = batch_stride;
	input_view_strides.at(dim + 1) = in_channels_stride * groups;
	for (int64_t i{0}; i < dim; ++i) {
		// stride of the output
		input_view_strides.at(i + 1) = input_stride.at(i) * stride.at(i);
		// stride inside the window
		input_view_strides.at(i + dim + 2) = input_stride.at(i) * dilation.at(i);
	}

	std::vector<torch::Tensor> results(groups);
	for (int64_t group_id{0}; group_id < groups; ++group_id) {
		int64_t const offset{in_channels_stride * group_id};
		auto const input_view{input.as_strided(input_view_sizes, input_view_strides, offset)};
		auto const weight_view{weight.narrow(0, group_id * out_channels_group, out_channels_group)};

		// input channels and all dimensions of the kernel weight will be flattened into a single
		// one dimensional kernel
		// result has sizes: N x output_size.. x out_channels
		auto result_group = hxtorch::matmul(
		    input_view.flatten(dim + 1), // N x output_size.. x (in_channels * prod(kernel_size..))
		    weight_view.flatten(1).t(),  // (in_channels * prod(kernel_size..)) x out_channels
		    num_sends, wait_between_events, mock);

		// moves out_channels dimension to get N x out_channels x output_size..
		results.at(group_id) = result_group.unsqueeze(1).transpose(1, -1).squeeze(-1);
	}
	auto result = torch::cat(results, 1).contiguous();

	if (bias) {
		std::vector<int64_t> bias_sizes(dim + 1, 1);
		bias_sizes.at(0) = -1;
		result = result.add(bias.value().view(bias_sizes));
	}
	return result;
}

} // namespace hxtorch::detail
