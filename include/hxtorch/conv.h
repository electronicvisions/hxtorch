#pragma once
#include <array>

#include <torch/torch.h>

namespace hxtorch {

torch::Tensor conv(
    torch::Tensor const& input,
    torch::Tensor const& weights,
    c10::optional<torch::Tensor> const& bias,
    std::vector<int64_t> stride,
    int64_t num_sends,
    int64_t wait_between_events,
    bool mock);

torch::Tensor conv1d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    int64_t stride = 1,
    int64_t num_sends = 1,
    int64_t wait_between_events = 25,
    bool mock = false);

torch::Tensor conv1d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    std::array<int64_t, 1> stride,
    int64_t num_sends = 1,
    int64_t wait_between_events = 25,
    bool mock = false);

torch::Tensor conv2d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    int64_t stride = 1,
    int64_t num_sends = 1,
    int64_t wait_between_events = 25,
    bool mock = false);

torch::Tensor conv2d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    std::array<int64_t, 2> stride,
    int64_t num_sends = 1,
    int64_t wait_between_events = 25,
    bool mock = false);

} // namespace hxtorch