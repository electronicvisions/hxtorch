#pragma once
#include <string>
#include <torch/torch.h>

namespace hxtorch {

namespace detail {
struct InferenceTracer;
} // namespace detail

/**
 * Inference tracer for a linear sequence of operations.
 * The traced operations' state is saved as a grenade::compute::Sequence, which can be
 * executed as a single operation without transformation to and from PyTorch tensors.
 * It is ensured, that no untraced modifications are made in-between traced operations by comparing
 * the last traced operation's output with the currently traced operation's input value.
 * @note Not final API or implementation, see Issue #3694
 */
class InferenceTracer
{
public:
	/**
	 * Construct inference tracer with filename to store traced operations to.
	 */
	InferenceTracer(std::string const& filename);

	/**
	 * Start tracing operations by registering tracer.
	 */
	void start();

	/**
	 * Stop tracing operations by deregistering tracer and save traced operations to given file.
	 * @return List of traced operation names
	 */
	std::vector<std::string> stop();

private:
	std::string m_filename;
	std::shared_ptr<detail::InferenceTracer> m_impl;
};


/**
 * Execute inference of stored trace.
 * @param input Input data to use
 * @param filename Filename to serialized operation trace
 */
torch::Tensor inference_trace(torch::Tensor const& input, std::string const& filename);

} // namespace hxtorch
