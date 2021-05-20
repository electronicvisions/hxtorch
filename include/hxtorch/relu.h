#pragma once
#include <torch/torch.h>

namespace hxtorch {

/**
 * Rectified linear unit operating on int8 value range.
 *
 * @param input Input tensor
 * @param mock Enable mock mode
 */
torch::Tensor relu(torch::Tensor const& input, bool mock = false);

/**
 * Rectified linear unit operating on int8 value range converting to uint5
 * value range.
 * The result is bit-shifted by @p shift after applying the ReLU and clipped
 * to the input range of BrainScaleS-2.
 *
 * @param input Input tensor
 * @param shift Amount of bits to shift before clipping
 * @param mock Enable mock mode
 */
torch::Tensor converting_relu(torch::Tensor const& input, int64_t shift = 2, bool mock = false);

} // namespace hxtorch
