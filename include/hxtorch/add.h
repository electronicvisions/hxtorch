#pragma once
#include <torch/torch.h>

namespace hxtorch {

/**
 * Elementwise addition operating on int8 value range.
 * @param input Input tensor
 * @param other Other tensor, which must be broadcastable to input tensor dimension
 * @param alpha The scalar multiplier for other
 * @param mock Whether to mock the hardware operation
 */
torch::Tensor add(
    torch::Tensor const& input, torch::Tensor const& other, double alpha = 1., bool mock = false);

} // namespace hxtorch