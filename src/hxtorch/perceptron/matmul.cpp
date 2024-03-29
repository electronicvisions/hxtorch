#include "hxtorch/perceptron/matmul.h"

#include "hxtorch/perceptron/mac.h"

#include <torch/torch.h>

#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/script.h>

namespace hxtorch::perceptron {

torch::Tensor matmul(
    torch::Tensor const& input,
    torch::Tensor const& other,
    int64_t const num_sends,
    int64_t wait_between_events,
    bool mock,
    int64_t madc_recording_neuron_id,
    std::string madc_recording_path)
{
	size_t const dim1 = input.dim();
	size_t const dim2 = other.dim();

	if (dim1 == 0 || dim2 == 0) {
		throw std::runtime_error("both arguments to matmul need to be at least 1D");
	}

	// add dimensions to get >= 2D
	torch::Tensor t1 = (dim1 == 1) ? input.unsqueeze(0) : input;
	torch::Tensor t2 = (dim2 == 1) ? other.unsqueeze(-1) : other;

	torch::Tensor res;
	if (dim1 <= 2 && dim2 <= 2) {
		res =
		    mac(t1, t2, num_sends, wait_between_events, mock, madc_recording_neuron_id,
		        madc_recording_path);
	} else if (dim1 >= 3 && dim2 <= 2) {
		// fold input's batch into its leading matrix dimension
		auto size1 = t1.sizes();
		auto size2 = t2.sizes();
		std::vector<int64_t> output_size;
		output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
		if (dim2 == 2) {
			output_size.push_back(size2[dim2 - 1]);
		}
		t1 = t1.reshape({-1, size1[size1.size() - 1]});
		res = mac(t1, t2, num_sends, wait_between_events, mock, madc_recording_neuron_id,
		          madc_recording_path)
		          .view(output_size);
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

static auto registry = torch::RegisterOperators().op("hxtorch_perceptron::matmul", &matmul);

} // namespace hxtorch::perceptron
