#pragma once

#include <torch/torch.h>

namespace hxtorch::perceptron::detail {

/**
 * Returns the output size of a convolution with given input size, kernel size, stride and dilation.
 */
int64_t conv1d_output_size(
    int64_t input_size, int64_t kernel_size, int64_t stride = 1, int64_t dilation = 1);

torch::Tensor expanded_conv1d(
    torch::Tensor const& input,
    torch::Tensor const& weights,
    c10::optional<torch::Tensor> const& bias,
    int64_t stride,
    int64_t dilation,
    int64_t num_expansions,
    int64_t num_sends,
    int64_t wait_between_events,
    bool mock);

} // namespace hxtorch::perceptron::detail
