#pragma once

#include <torch/torch.h>

namespace hxtorch::perceptron::detail {

/**
 * Calculate forward-pass of multiply accumulate operation.
 * Input dimensions supported are 1D or 2D, where in the latter the input plane is the highest
 * dimension and the first dimension describes which input vector to choose.
 * The multiply accumulate therefore multiplies the last input dimension with the first weights
 * dimension like y = x^T W.
 * @param x Input (1D or 2D)
 * @param weights 2D weight matrix
 * @param num_sends How often to send the (same) input vector
 * @param madc_recording_neuron_id Neuron ID to record via MADC
 * @param madc_recording_path Path to which to store MADC neuron membrane recordings. If file exists
 * new data is appended. By default recording is disabled.
 * @return Resulting tensor
 */
torch::Tensor mac_forward(
    torch::Tensor x,
    torch::Tensor weights,
    int64_t num_sends,
    int64_t wait_between_events,
    int64_t madc_recording_neuron_id,
    std::string madc_recording_path);

/**
 * Mocks the forward-pass of the multiply accumulate operation.
 * Input dimensions supported are 1D or 2D, where in the latter the input plane is the highest
 * dimension and the first dimension describes which input vector to choose.
 * The multiply accumulate therefore multiplies the last input dimension with the first weights
 * dimension like y = x^T W.
 * @param x Input (1D or 2D)
 * @param weights 2D weight matrix
 * @param num_sends How often to send the (same) input vector
 * @return Resulting tensor
 */
torch::Tensor mac_mock_forward(
    torch::Tensor const& x, torch::Tensor const& weights, int64_t num_sends);

torch::autograd::variable_list mac_backward(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor weights);

} // namespace hxtorch::perceptron::detail
