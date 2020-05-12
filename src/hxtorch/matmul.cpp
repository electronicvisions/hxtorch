#include "hxtorch/matmul.h"

#include "hxtorch/mac.h"

#include <torch/torch.h>

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/script.h>

namespace hxtorch {

torch::Tensor matmul(
    torch::Tensor const& tensor1,
    torch::Tensor const& tensor2,
    int64_t const num_sends,
    int64_t wait_between_events,
    bool mock)
{
	size_t const dim1 = tensor1.dim();
	size_t const dim2 = tensor2.dim();

	if (dim1 == 0 || dim2 == 0) {
		throw std::runtime_error("both arguments to matmul need to be at least 1D");
	}

	// add dimensions to get >= 2D
	torch::Tensor t1 = (dim1 == 1) ? tensor1.unsqueeze(0) : tensor1;
	torch::Tensor t2 = (dim2 == 1) ? tensor2.unsqueeze(-1) : tensor2;

	torch::Tensor res;
	if (dim1 <= 2 && dim2 <= 2) {
		res = mac(t1, t2, num_sends, wait_between_events, mock);
	} else if (dim1 >= 3 && dim2 <= 2) {
		// fold tensor1's batch into its leading matrix dimension
		auto size1 = t1.sizes();
		auto size2 = t2.sizes();
		std::vector<int64_t> output_size;
		output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
		if (dim2 == 2) {
			output_size.push_back(size2[dim2 - 1]);
		}
		t1 = t1.reshape({-1, size1[size1.size() - 1]});
		res = mac(t1, t2, num_sends, wait_between_events, mock).view(output_size);
	} else {
		// TODO: implement >2D weights
		// TODO: implement batched mode
		throw std::runtime_error(
		    "matmul for " + std::to_string(dim1) + "D and " + std::to_string(dim2) +
		    "D inputs is not implemented yet :(");
	}

	// remove added dimensions
	if (dim1 == 1) {
		res = res.squeeze(-2);
	}
	if (dim2 == 1) {
		res = res.squeeze(-1);
	}
	return res;
}

static auto registry = torch::RegisterOperators().op("hxtorch::matmul", &matmul);

} // namespace hxtorch
