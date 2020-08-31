#pragma once
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>
#include <torch/torch.h>

#include "grenade/vx/compute_sequence.h"

namespace hxtorch::detail {

/**
 * Inference tracer implementation.
 * Currently only traces operation names.
 */
struct InferenceTracer
{
	std::vector<std::string> operation_names;
	grenade::vx::ComputeSequence ops;

	/**
	 * Check input is equal to saved output.
	 * Allows to ensure that between two traecd operations, the data is modified only by an identity
	 * function and thus all not-traced operations in between can be discarded.
	 * @param value Value to check
	 */
	void check_input(torch::Tensor const& value) const;

	/**
	 * Update cached output value.
	 * This is compared with the input of the next traced operation to ensure no modifications
	 * in-between.
	 * @param value Value to update
	 */
	void update_output(torch::Tensor const& value);

private:
	/**
	 * Last output tensor to compare to next operations input tensor for sanity check.
	 */
	std::optional<torch::Tensor> m_last_output;
};

/**
 * Get singleton set of registered inference tracers.
 */
std::unordered_set<std::shared_ptr<InferenceTracer>>& getInferenceTracer();

/**
 * Check whether inference tracers are registered.
 * @return Boolean value
 */
bool has_tracer();

/**
 * Check all tracers for equality of the output of the last traced operation with the given value.
 * @throws std::runtime_error On this operations input being unequal to last operations output
 * @param value Value to check
 */
void tracer_check_input(torch::Tensor const& value);

/**
 * Update all tracers' output of the last traced operation with the given value.
 * @param value Value to update
 */
void tracer_update_output(torch::Tensor const& value);

/**
 * Add operation to trace.
 * @param name Name to use
 * @param op Operation to add
 */
void tracer_add(std::string const& name, grenade::vx::ComputeSequence::Entry&& op);

} // namespace hxtorch::detail
