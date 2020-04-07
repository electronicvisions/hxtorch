#pragma once

#include <torch/torch.h>

namespace hxtorch::detail {

/**
 * Calculate forward-pass of multiply accumulate operation.
 * Input dimensions supported are 1D or 2D, where in the latter the input plane is the highest
 * dimension and the first dimension describes which input vector to choose.
 * The multiply accumulate therefore multiplies the last input dimension with the first weights
 * dimension like y = x^T W.
 * @param x Input (1D or 2D)
 * @param weights 2D weight matrix
 * @param num_sends How often to send the (same) input vector
 * @return Resulting tensor
 */
torch::Tensor mac_forward(
    torch::Tensor x, torch::Tensor weights, int64_t num_sends, int64_t wait_between_events);

torch::autograd::variable_list mac_backward(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor weights);

} // namespace hxtorch::detail
