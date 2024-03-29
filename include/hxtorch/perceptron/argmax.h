#pragma once
#include <torch/torch.h>

namespace hxtorch::perceptron {

/**
 * Arg max operation on int8 value range.
 *
 * @param input The input tensor
 * @param dim The dimension to reduce. If unspecified, the argmax of the flattened
 *            input is returned.
 * @param keepdim Whether the output tensor has @p dim retained or not. Ignored
 *                if @p dim is unspecified.
 * @param mock Enable mock mode
 *
 * @return The indices of the maximum values of a tensor across a dimension
 */
torch::Tensor argmax(
    torch::Tensor const& input,
    c10::optional<int64_t> dim = c10::nullopt,
    bool keepdim = false,
    bool mock = false);

} // namespace hxtorch::perceptron
