#pragma once
#include <torch/torch.h>

#include "hxtorch/perceptron/constants.h"

namespace hxtorch::perceptron {

using namespace hxtorch::perceptron::constants::defaults;

/**
 * Drop-in replacement for the torch.matmul operation that uses BrainScaleS-2.
 *
 * @note
 * The current implementation only supports @p other to be 1D or 2D.
 *
 * @param input First input tensor
 * @param other Second input tensor
 * @param num_sends How often to send the (same) input vector
 * @param wait_between_events How long to wait (in FPGA cycles) between events
 * @param mock: Enable mock mode
 *
 * @return Resulting tensor
 */
torch::Tensor matmul(
    torch::Tensor const& input,
    torch::Tensor const& other,
    int64_t num_sends = 1,
    int64_t wait_between_events = wait_between_events,
    bool mock = false);

} // namespace hxtorch::perceptron
