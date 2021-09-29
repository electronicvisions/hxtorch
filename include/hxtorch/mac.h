#pragma once
#include <torch/torch.h>

#include "hxtorch/constants.h"

namespace hxtorch {

class MAC : public torch::autograd::Function<MAC>
{
public:
	static torch::autograd::variable_list forward(
	    torch::autograd::AutogradContext* ctx,
	    torch::autograd::Variable x,
	    torch::autograd::Variable weights,
	    int64_t num_sends,
	    int64_t wait_between_events,
	    bool mock);

	static torch::autograd::variable_list backward(
	    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output);
};

using namespace hxtorch::constants::defaults;

/**
 * The bare mutliply-accumulate operation of BrainScaleS-2. A 1D input @p x
 * is multiplied by the weight matrix @p weights. If @p x is two-dimensional,
 * the weights are sent only once to the synapse array and the inputs are
 * consecutively multiplied as a 1D vector.
 *
 * @param x Input tensor
 * @param weights The weights of the synapse array
 * @param num_sends How often to send the (same) input vector
 * @param wait_between_events How long to wait (in FPGA cycles) between events
 * @param mock Enable mock mode
 *
 * @return Resulting tensor
 */
torch::Tensor mac(
    torch::Tensor const& x,
    torch::Tensor const& weights,
    int64_t num_sends = 1,
    int64_t wait_between_events = wait_between_events,
    bool mock = false);

} // namespace hxtorch
