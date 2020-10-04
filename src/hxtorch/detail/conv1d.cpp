#include "hxtorch/detail/conv1d.h"
#include "hxtorch/detail/conv.h"

#include <torch/torch.h>

namespace F = torch::nn::functional;

namespace hxtorch::detail {

size_t conv1d_num_outputs(size_t x_size, size_t weights_size, size_t stride)
{
	return ((x_size - (weights_size - 1) - 1) / stride) + 1;
}

torch::Tensor expanded_conv1d(
    torch::Tensor const& input,
    torch::Tensor const& weight,
    c10::optional<torch::Tensor> const& bias,
    int64_t const stride,
    int64_t const num_expansions,
    int64_t const num_sends,
    int64_t const wait_between_events,
    bool const mock)
{
	auto const weight_sizes = weight.sizes();
	auto const input_sizes = input.sizes();
	auto weight_new = torch::zeros(
	    {weight_sizes.at(0) * num_expansions /* out_channels */,
	     weight_sizes.at(1) /* in_channels / groups */,
	     weight_sizes.at(2) + stride * (num_expansions - 1) /* kernel_size */},
	    input.dtype());
	for (int64_t i = 0; i < num_expansions; ++i) {
		weight_new.index_put_(
		    {torch::indexing::Slice(i, torch::indexing::None, num_expansions),
		     torch::indexing::Slice(),
		     torch::indexing::Slice(i * stride, i * stride + weight_sizes.at(2))},
		    weight);
	}
	c10::optional<torch::Tensor> bias_new;
	if (bias) {
		bias_new = c10::optional<torch::Tensor>(bias.value().repeat_interleave(num_expansions));
	}
	int64_t stride_new = stride * num_expansions;

	// compute output length and pad the input with zeros if needed
	int64_t l_out = ((input_sizes.at(2) - weight_sizes.at(2)) / stride) + 1;
	int64_t l_out_new =
	    (((input_sizes.at(2) - weight_new.sizes().at(2)) / stride_new) + 1) * num_expansions;
	torch::Tensor input_new;
	if ((l_out > l_out_new) || (input_sizes.at(2) < weight_new.sizes().at(2))) {
		input_new = F::pad(input, F::PadFuncOptions({0, stride_new}));
	} else {
		input_new = input;
	}

	auto out =
	    conv(input_new, weight_new, bias_new, {stride_new}, num_sends, wait_between_events, mock)
	        .permute({0, 2, 1})
	        .reshape({input_sizes.at(0), -1, weight_sizes.at(0), num_expansions})
	        .permute({0, 2, 1, 3})
	        .reshape({input_sizes.at(0), weight_sizes.at(0), -1});
	return out.index({"...", torch::indexing::Slice(torch::indexing::None, l_out)}).contiguous();
}

} // namespace hxtorch::detail
