#include "hxtorch/argmax.h"

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "hxtorch/detail/argmax.h"

namespace hxtorch {

torch::Tensor argmax(
    torch::Tensor const& input,
    c10::optional<int64_t> const dim,
    bool const keepdim,
    bool const mock)
{
	return mock ? detail::argmax_mock(input, dim, keepdim) : detail::argmax(input, dim, keepdim);
}

static auto registry = torch::RegisterOperators().op("hxtorch::argmax", &argmax);

} // namespace hxtorch
