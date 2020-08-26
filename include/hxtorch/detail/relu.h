#include <torch/torch.h>

namespace hxtorch::detail {

torch::Tensor relu_mock_forward(torch::Tensor const& input);

torch::Tensor relu_forward(torch::Tensor const& input);

torch::autograd::variable_list relu_backward(
    torch::Tensor const& grad_output, torch::Tensor const& input);

} // namespace hxtorch::detail
