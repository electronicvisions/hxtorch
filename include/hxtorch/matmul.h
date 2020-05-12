#pragma once
#include <torch/torch.h>

namespace hxtorch {

/**
 * Drop-in replacement for the torch.matmul operation that uses HICANN-X.
 * The current implementation only supports tensor2 to be 1D or 2D.
 * @param tensor1 First input tensor, allowed range [0, 31]
 * @param tensor2 Second input tensor, allowed range: [-63, 63]
 * @param num_sends How often to send the (same) input vector
 * @param wait_between_events How long to wait (in FPGA cycles) between events
 * @return Resulting tensor
 */
torch::Tensor matmul(
    torch::Tensor const& tensor1,
    torch::Tensor const& tensor2,
    int64_t num_sends = 1,
    int64_t wait_between_events = 25,
    bool mock = false);

} // namespace hxtorch
