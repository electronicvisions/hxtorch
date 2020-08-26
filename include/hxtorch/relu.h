#pragma once
#include <torch/torch.h>

namespace hxtorch {

/**
 * Rectified linear unit operating on int8 value range.
 * @param input Input tensor
 * @param mock Whether to mock the hardware operation
 */
torch::Tensor relu(torch::Tensor const& input, bool mock = false);

/**
 * Rectified linear unit operating on int8 value range converting to uint5 value range.
 * @param input Input tensor
 * @param shift Amount of bits to shift before clamping
 * @param mock Whether to mock the hardware operation
 */
torch::Tensor converting_relu(torch::Tensor const& input, int64_t shift = 2, bool mock = false);

} // namespace hxtorch
