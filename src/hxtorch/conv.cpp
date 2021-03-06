#include "hxtorch/conv.h"

#include "hxtorch/detail/conv.h"

#include <torch/torch.h>

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
//#include <torch/custom_class.h>
//#include <torch/extension.h>
#include <torch/script.h>

namespace hxtorch {

torch::Tensor conv1d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    int64_t const stride,
    int64_t const num_sends,
    int64_t const wait_between_events,
    bool const mock)
{
	return detail::conv(input, weight, bias, {stride}, num_sends, wait_between_events, mock);
}

torch::Tensor conv1d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    std::array<int64_t, 1> const stride,
    int64_t const num_sends,
    int64_t const wait_between_events,
    bool const mock)
{
	return detail::conv(input, weight, bias, {stride[0]}, num_sends, wait_between_events, mock);
}

torch::Tensor conv2d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    int64_t const stride,
    int64_t const num_sends,
    int64_t const wait_between_events,
    bool const mock)
{
	return detail::conv(
	    input, weight, bias, {stride, stride}, num_sends, wait_between_events, mock);
}

torch::Tensor conv2d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    std::array<int64_t, 2> const stride,
    int64_t const num_sends,
    int64_t const wait_between_events,
    bool const mock)
{
	return detail::conv(
	    input, weight, bias, {stride[0], stride[1]}, num_sends, wait_between_events, mock);
}

} // namespace hxtorch
