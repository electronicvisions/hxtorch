#pragma once
#include <torch/torch.h>

namespace hxtorch {

/**
 * Rectified linear unit operating on int8 value range.
 * @param input Input tensor
 * @param mock Whether to mock the hardware operation
 */
torch::Tensor relu(torch::Tensor const& input, bool mock = false);

} // namespace hxtorch
