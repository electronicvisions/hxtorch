#pragma once
#include <array>

#include <torch/torch.h>

namespace hxtorch::perceptron::detail {

/**
 * Returns the output size of a convolution with given input size, kernel size, stride and dilation.
 */
std::vector<int64_t> conv_output_size(
    std::vector<int64_t> input_size,
    std::vector<int64_t> kernel_size,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation);

torch::Tensor conv(
    torch::Tensor const& input,
    torch::Tensor const& weights,
    c10::optional<torch::Tensor> const& bias,
    std::vector<int64_t> const& stride,
    std::vector<int64_t> const& dilation,
    int64_t num_sends,
    int64_t wait_between_events,
    bool mock);

} // namespace hxtorch::perceptron::detail
