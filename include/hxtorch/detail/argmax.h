#include <torch/torch.h>

namespace hxtorch::detail {

torch::Tensor argmax_mock(
    torch::Tensor const& input, c10::optional<int64_t> dim = c10::nullopt, bool keepdim = false);

torch::Tensor argmax(
    torch::Tensor const& input, c10::optional<int64_t> dim = c10::nullopt, bool keepdim = false);

} // namespace hxtorch::detail
