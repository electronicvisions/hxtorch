#pragma once

#include <torch/torch.h>

namespace hxtorch::detail {

size_t conv1d_num_outputs(size_t x_size, size_t weights_size, size_t stride);

torch::Tensor expanded_conv1d(
    torch::Tensor const& input,
    torch::Tensor const& weights,
    c10::optional<torch::Tensor> const& bias,
    int64_t stride,
    int64_t num_expansions,
    int64_t num_sends,
    int64_t wait_between_events,
    bool mock);

} // namespace hxtorch::detail
