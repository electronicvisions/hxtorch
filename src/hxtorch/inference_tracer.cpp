#include "hxtorch/inference_tracer.h"

#include "hxtorch/detail/connection.h"
#include "hxtorch/detail/conversion.h"
#include "hxtorch/detail/inference_tracer.h"

#include <fstream>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>

namespace hxtorch {

InferenceTracer::InferenceTracer(std::string const& filename) : m_filename(filename), m_impl() {}

void InferenceTracer::start()
{
	m_impl = std::make_shared<detail::InferenceTracer>();
	detail::getInferenceTracer().insert(m_impl);
}

std::vector<std::string> InferenceTracer::stop()
{
	assert(m_impl);
	detail::getInferenceTracer().erase(m_impl);

	{
		std::ofstream file(m_filename);
		{
			cereal::BinaryOutputArchive oa(file);
			oa(m_impl->ops);
		}
	}

	auto const ret = m_impl->operation_names;
	m_impl.reset();
	return ret;
}


namespace {

template <typename T>
std::tuple<std::vector<std::vector<T>>, std::vector<int64_t>> convert_inference_trace_input(
    torch::Tensor const& input)
{
	detail::tracer_check_input(input);

	if (input.dim() > 2) {
		throw std::runtime_error("inference_trace can only operate on 1D or 2D input.");
	}

	auto const sizes = input.sizes().vec();

	auto input_2d = input.reshape({-1, sizes.at(sizes.size() - 1)});
	auto const sizes_2d = input_2d.sizes().vec();

	grenade::vx::IODataList::Entry input_variant;
	auto input_a = input_2d.accessor<float, 2>();
	if constexpr (std::is_same_v<T, grenade::vx::Int8>) {
		std::vector<std::vector<grenade::vx::Int8>> input_in(sizes_2d.at(0));
		for (int64_t i = 0; i < sizes_2d.at(0); ++i) {
			input_in[i].resize(sizes_2d.at(1));
			for (int64_t j = 0; j < sizes_2d.at(1); ++j) {
				input_in[i][j] = grenade::vx::Int8(input_a[i][j]);
			}
		}
		return {input_in, sizes_2d};
	} else {
		std::vector<std::vector<grenade::vx::UInt5>> input_in(sizes_2d.at(0));
		for (int64_t i = 0; i < sizes_2d.at(0); ++i) {
			input_in[i].resize(sizes_2d.at(1));
			for (int64_t j = 0; j < sizes_2d.at(1); ++j) {
				input_in[i][j] = detail::convert_activation(input_a[i][j]);
			}
		}
		return {input_in, sizes_2d};
	}
}

template <typename T>
torch::Tensor convert_inference_trace_output(
    std::vector<std::vector<T>> const& results,
    std::vector<int64_t> const& sizes_2d,
    std::vector<int64_t> const& sizes)
{
	auto ret = torch::zeros(sizes_2d);
	auto ret_a = ret.accessor<float, 2>();
	for (int64_t i = 0; i < sizes_2d.at(0); ++i) {
		for (int64_t j = 0; j < sizes_2d.at(1); ++j) {
			ret_a[i][j] = static_cast<float>(results[i][j]);
		}
	}
	ret = ret.reshape(sizes);
	detail::tracer_update_output(ret);
	return ret;
}

} // namespace

torch::Tensor inference_trace(torch::Tensor const& input, std::string const& filename)
{
	grenade::vx::ComputeSequence ops;
	{
		std::ifstream file(filename);
		{
			cereal::BinaryInputArchive ia(file);
			ia(ops);
		}
	}

	if (ops.data.empty()) {
		throw std::runtime_error("Empty trace can't be run.");
	}

	grenade::vx::IODataList::Entry input_variant;
	std::vector<int64_t> sizes_2d;
	if (std::holds_alternative<grenade::vx::ComputeSingleMAC>(ops.data.front())) {
		auto const [i, s] = convert_inference_trace_input<grenade::vx::UInt5>(input);
		input_variant = i;
		sizes_2d = s;
	} else {
		auto const [i, s] = convert_inference_trace_input<grenade::vx::Int8>(input);
		input_variant = i;
		sizes_2d = s;
	}

	if (!hxtorch::detail::getConnection()) {
		throw std::runtime_error("No connection allocated.");
	}
	auto const result_variant =
	    ops.run(input_variant, hxtorch::detail::getChip(), *hxtorch::detail::getConnection());

	torch::Tensor ret;
	if (std::holds_alternative<grenade::vx::ComputeSingleConvertingReLU>(ops.data.front())) {
		ret = convert_inference_trace_output(
		    std::get<std::vector<std::vector<grenade::vx::UInt5>>>(result_variant), sizes_2d,
		    input.sizes().vec());
	} else {
		ret = convert_inference_trace_output(
		    std::get<std::vector<std::vector<grenade::vx::Int8>>>(result_variant), sizes_2d,
		    input.sizes().vec());
	}
	return ret;
}

} // namespace hxtorch
