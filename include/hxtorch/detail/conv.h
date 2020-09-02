#pragma once
#include <array>

#include <torch/torch.h>

namespace hxtorch::detail {

std::vector<int64_t> conv_num_outputs(
    std::vector<int64_t> input_size, std::vector<int64_t> kernel_size, std::vector<int64_t> stride);

torch::Tensor conv_fold_input(
    torch::Tensor const& input, std::vector<int64_t> kernel_size, std::vector<int64_t> stride);

torch::Tensor conv_unfold_input(
    torch::Tensor const& input,
    size_t in_channels,
    std::vector<int64_t> kernel_size,
    std::vector<int64_t> num_cols,
    std::vector<int64_t> stride);

torch::Tensor conv_fold_kernel(torch::Tensor const& kernel);

torch::Tensor conv_unfold_kernel(
    torch::Tensor const& kernel, std::vector<int64_t> kernel_size, int64_t in_channels);

torch::Tensor conv_permute_output(torch::Tensor const& value);

torch::Tensor conv_unpermute_output(torch::Tensor const& value);

torch::Tensor conv_unfold_output(torch::Tensor const& value, std::vector<int64_t> num_outputs);

torch::Tensor conv_fold_output(torch::Tensor const& value);

torch::Tensor conv(
    torch::Tensor const& input,
    torch::Tensor const& weights,
    c10::optional<torch::Tensor> const& bias,
    std::vector<int64_t> const& stride,
    int64_t num_sends,
    int64_t wait_between_events,
    bool mock);

} // namespace hxtorch::detail
