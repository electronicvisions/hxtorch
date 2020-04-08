#include "hxtorch/detail/conv.h"

#include "hxtorch/detail/conv1d.h"
#include "hxtorch/detail/iterator.h"
#include "hxtorch/detail/mac.h"
#include "hxtorch/detail/narrow.h"

#include <torch/torch.h>

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/script.h>

namespace hxtorch::detail {

std::vector<int64_t> conv_num_outputs(
    std::vector<int64_t> input_size, std::vector<int64_t> kernel_size, std::vector<int64_t> stride)
{
	if (input_size.size() != kernel_size.size() || input_size.size() != stride.size()) {
		throw std::runtime_error("conv_num_outputs: sizes of inputs don't match.");
	}
	auto const dim = input_size.size();
	std::vector<int64_t> num_outputs(dim);
	for (size_t i = 0; i < dim; ++i) {
		num_outputs.at(i) = conv1d_num_outputs(input_size.at(i), kernel_size.at(i), stride.at(i));
	}
	return num_outputs;
}

torch::Tensor conv_fold_input(
    torch::Tensor const& input, std::vector<int64_t> kernel_size, std::vector<int64_t> stride)
{
	if (kernel_size.size() != stride.size()) {
		throw std::runtime_error("kernel_size and stride shape don't match.");
	}
	if (input.dim() != static_cast<int64_t>(2 /* minibatch + in_channels */ + kernel_size.size())) {
		throw std::runtime_error("conv_fold_input input dimension does not match.");
	}

	auto const kernel_dim = kernel_size.size();

	auto const input_tensor_sizes = input.sizes().vec();
	auto const input_sizes =
	    std::vector<int64_t>(input_tensor_sizes.end() - kernel_dim, input_tensor_sizes.end());

	auto const num_cols = conv_num_outputs(input_sizes, kernel_size, stride);

	std::vector<int64_t> kernel_dims(kernel_dim);
	for (size_t i = 0; i < kernel_dim; ++i) {
		kernel_dims.at(i) = input.dim() - (kernel_dim - i);
	}

	torch::Tensor input_folded = input.contiguous().narrow(kernel_dims.at(0), 0, kernel_size.at(0));
	for (size_t i = 1; i < kernel_dim; ++i) {
		input_folded = input_folded.narrow(kernel_dims.at(i), 0, kernel_size.at(i));
	}
	auto input_folded_sizes = input_folded.sizes().vec();
	input_folded_sizes.insert(input_folded_sizes.end() - kernel_dim - 1 /* in_channels */, 1);
	input_folded = input_folded.reshape(input_folded_sizes);

	auto it = MultidimIterator(num_cols);
	++it; // first location already present
	for (; it != it.end(); ++it) {
		auto const location = *it;
		std::vector<int64_t> offset(kernel_dim);
		for (size_t i = 0; i < kernel_dim; ++i) {
			offset.at(i) = location.at(i) * stride.at(i);
		}
		auto const local_input_folded =
		    multi_narrow(input, kernel_dims, offset, kernel_size).reshape(input_folded_sizes);
		input_folded =
		    torch::cat({input_folded, local_input_folded}, input_folded.dim() - kernel_dim - 2);
	}
	auto input_folded_sizes_flat = input_folded.sizes().vec();
	int64_t kernel_size_flat = 1;
	for (auto const k : kernel_size) {
		kernel_size_flat *= k;
	}
	for (size_t i = 0; i < kernel_dim + 1 /* in_channels */; ++i) {
		input_folded_sizes_flat.pop_back();
	}
	input_folded_sizes_flat.push_back(
	    kernel_size_flat * input.sizes().vec().at(1) /* in_channels */);
	auto const input_folded_flat = input_folded.reshape(input_folded_sizes_flat);
	return input_folded_flat;
}

torch::Tensor conv_unfold_input(
    torch::Tensor const& input,
    size_t in_channels,
    std::vector<int64_t> kernel_size,
    std::vector<int64_t> num_cols,
    std::vector<int64_t> stride)
{
	if (kernel_size.size() != num_cols.size() || kernel_size.size() != stride.size()) {
		throw std::runtime_error("kernel_size, num_cols or stride shape don't match.");
	}
	if (input.dim() != 3 /* minibatch + flattening + 1D-input */) {
		throw std::runtime_error("conv_unfold_input input dimension does not match.");
	}
	auto const kernel_dim = kernel_size.size();

	auto input_kernel_unflattened_sizes = input.sizes().vec();
	input_kernel_unflattened_sizes.pop_back();
	input_kernel_unflattened_sizes.push_back(in_channels);
	for (auto const k : kernel_size) {
		input_kernel_unflattened_sizes.push_back(k);
	}
	auto const input_kernel_unflattened = input.reshape(input_kernel_unflattened_sizes);

	std::vector<int64_t> input_size(kernel_dim);
	for (size_t i = 0; i < kernel_dim; ++i) {
		input_size.at(i) = (num_cols.at(i) - 1) * stride.at(i) + kernel_size.at(i);
	}

	auto input_sizes = input.sizes().vec();
	input_sizes.pop_back();
	input_sizes.pop_back();
	input_sizes.push_back(in_channels);
	for (size_t i = 0; i < kernel_dim; ++i) {
		input_sizes.push_back(input_size.at(i));
	}

	torch::Tensor input_unfolded = torch::zeros(input_sizes, input.dtype());

	std::vector<int64_t> dims;
	std::vector<int64_t> lengths;
	for (size_t i = 0; i < kernel_dim; ++i) {
		dims.push_back(input_unfolded.dim() - (kernel_dim - i));
		lengths.push_back(kernel_size.at(i));
	}
	size_t location_linear = 0;
	for (auto it = MultidimIterator(num_cols); it != it.end(); ++it) {
		auto location = *it;
		auto const input_view = input_kernel_unflattened.narrow(1, location_linear, 1).squeeze(1);
		std::vector<int64_t> offsets;
		for (size_t i = 0; i < kernel_dim; ++i) {
			offsets.push_back(location.at(i) * stride.at(i));
		}
		multi_narrow(input_unfolded, dims, offsets, lengths) = input_view;
		location_linear++;
	}
	return input_unfolded;
}

torch::Tensor conv_fold_kernel(torch::Tensor const& kernel)
{
	if (kernel.dim() < 3) {
		throw std::runtime_error("conv_fold_kernel expects kernel of shape (out_channels, "
		                         "in_channels, (single_kernel)..)");
	}
	return kernel.reshape({kernel.sizes().vec().at(0), -1}).t();
}

torch::Tensor conv_unfold_kernel(
    torch::Tensor const& kernel, std::vector<int64_t> kernel_size, int64_t in_channels)
{
	if (kernel.dim() != 2) {
		throw std::runtime_error("conv_unfold_kernel expects kernel of 2D shape.");
	}
	auto const out_channels = kernel.sizes().vec().at(1);
	std::vector<int64_t> kernel_shape;
	kernel_shape.push_back(out_channels);
	kernel_shape.push_back(in_channels);
	std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(kernel_shape));
	return kernel.t().reshape(kernel_shape);
}

torch::Tensor conv_permute_output(torch::Tensor const& value)
{
	std::vector<int64_t> dims(value.dim());
	dims.at(0) = 0;
	dims.at(1) = dims.size() - 1;
	for (size_t i = 2; i < dims.size(); ++i) {
		dims.at(i) = i - 1;
	}
	return value.permute(dims);
}

torch::Tensor conv_unpermute_output(torch::Tensor const& value)
{
	std::vector<int64_t> dims(value.dim());
	dims.at(0) = 0;
	for (size_t i = 1; i < dims.size() - 1; ++i) {
		dims.at(i) = i + 1;
	}
	dims.back() = 1;
	return value.permute(dims);
}

torch::Tensor conv_unfold_output(torch::Tensor const& value, std::vector<int64_t> num_outputs)
{
	if (value.dim() != 3 /* minibatch + out_channels + folded */) {
		throw std::runtime_error("conv_unfold_output: provided tensor's dimension is not "
		                         "(minibatch, out_channels, folded).");
	}
	auto const kernel_dim = num_outputs.size();
	auto sizes = value.sizes().vec();
	sizes.resize(sizes.size() - 1 + kernel_dim);
	std::copy(num_outputs.begin(), num_outputs.end(), sizes.end() - kernel_dim);
	return value.reshape(sizes);
}

torch::Tensor conv_fold_output(torch::Tensor const& value)
{
	auto const kernel_dim = value.dim() - 2 /* minibatch + out_channels */;
	if (kernel_dim < 1) {
		throw std::runtime_error("conv_fold_output: provided tensor's dimension is too low.");
	}
	auto sizes = value.sizes().vec();
	int64_t output_linear_size = 1;
	for (int64_t i = 0; i < kernel_dim; ++i) {
		output_linear_size *= sizes.at(sizes.size() - i - 1);
	}
	sizes.resize(sizes.size() - kernel_dim + 1);
	sizes.back() = output_linear_size;
	return value.reshape(sizes);
}

} // namespace hxtorch::detail
