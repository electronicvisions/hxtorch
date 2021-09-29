#pragma once
#include <array>

#include <torch/torch.h>

#include "hxtorch/constants.h"

namespace hxtorch {

torch::Tensor conv1d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    int64_t stride = 1,
    int64_t num_sends = 1,
    int64_t wait_between_events = hxtorch::constants::defaults::wait_between_events,
    bool mock = false);

torch::Tensor conv1d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    std::array<int64_t, 1> stride,
    int64_t num_sends = 1,
    int64_t wait_between_events = hxtorch::constants::defaults::wait_between_events,
    bool mock = false);

/**
 * 1D convolution operation that unrolls the weight matrix for execution
 * on hardware. This maximizes the use of the synapses array.
 *
 * @note
 * Fixed-pattern noise cannot be individually compensated for during
 * training, because the same weights are used at different locations!
 *
 * @param input Input tensor of shape (minibatch, in_channels, *iW*)
 * @param weight Filters of shape (out_channels, in_channels / groups, *kW*)
 * @param bias Optional bias of shape (out_channels)
 * @param stride Stride of the convolving kernel
 * @param num_expansions Number of enrolled kernels that will be placed side
 *                       by side in a single operation
 * @param num_sends How often to send the (same) input vector
 * @param wait_between_events How long to wait (in FPGA cycles) between events
 * @param mock Enable mock mode
 */
torch::Tensor expanded_conv1d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    int64_t stride = 1,
    int64_t num_expansions = 1,
    int64_t num_sends = 1,
    int64_t wait_between_events = hxtorch::constants::defaults::wait_between_events,
    bool mock = false);

torch::Tensor expanded_conv1d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    std::array<int64_t, 1> stride,
    int64_t num_expansions = 1,
    int64_t num_sends = 1,
    int64_t wait_between_events = hxtorch::constants::defaults::wait_between_events,
    bool mock = false);

torch::Tensor conv2d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    int64_t stride = 1,
    int64_t num_sends = 1,
    int64_t wait_between_events = hxtorch::constants::defaults::wait_between_events,
    bool mock = false);

torch::Tensor conv2d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    std::array<int64_t, 2> stride,
    int64_t num_sends = 1,
    int64_t wait_between_events = hxtorch::constants::defaults::wait_between_events,
    bool mock = false);

} // namespace hxtorch
