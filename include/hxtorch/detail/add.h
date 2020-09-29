#include <torch/torch.h>

namespace hxtorch::detail {

torch::Tensor add_mock_forward(torch::Tensor const& input, torch::Tensor const& other);

torch::Tensor add_forward(torch::Tensor const& input, torch::Tensor const& other);

torch::autograd::variable_list add_backward(
    torch::Tensor const& grad_output, torch::Tensor const& input, torch::Tensor const& other);

} // namespace hxtorch::detail
