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
 * @param madc_recording_neuron_id Neuron ID to record via MADC
 * @param madc_recording_path Path to which to store MADC neuron membrane recordings in CSV format.
 * If file exists new data is appended. By default recording is disabled.
 * @throws std::runtime_error When MADC recording is enabled but mock-mode is used.
 *
 * @return Resulting tensor
 */
torch::Tensor matmul(
    torch::Tensor const& input,
    torch::Tensor const& other,
    int64_t num_sends = 1,
    int64_t wait_between_events = wait_between_events,
    bool mock = false,
    int64_t madc_recording_neuron_id = 0,
    std::string madc_recording_path = "");

} // namespace hxtorch::perceptron
