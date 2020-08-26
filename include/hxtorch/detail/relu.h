#include <torch/torch.h>

namespace hxtorch::detail {

torch::Tensor relu_mock_forward(torch::Tensor const& input);

torch::Tensor relu_forward(torch::Tensor const& input);

torch::autograd::variable_list relu_backward(
    torch::Tensor const& grad_output, torch::Tensor const& input);


torch::Tensor converting_relu_mock_forward(torch::Tensor const& input, int64_t shift);

torch::Tensor converting_relu_forward(torch::Tensor const& input, int64_t shift);

torch::autograd::variable_list converting_relu_backward(
    torch::Tensor const& grad_output, torch::Tensor const& input, int64_t shift);

} // namespace hxtorch::detail
