#pragma once
#include <torch/torch.h>

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


torch::Tensor mac(
    torch::Tensor const& x,
    torch::Tensor const& weights,
    int64_t num_sends = 1,
    int64_t wait_between_events = 5,
    bool mock = false);

} // namespace hxtorch
