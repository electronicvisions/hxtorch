#include "hxtorch/detail/inference_tracer.h"

#include <sstream>
#include <log4cxx/logger.h>

namespace hxtorch::detail {

void InferenceTracer::check_input(torch::Tensor const& value) const
{
	static log4cxx::LoggerPtr const logger =
	    log4cxx::Logger::getLogger("hxtorch.InferenceTracer.check_input");
	LOG4CXX_TRACE(logger, "Checking input value(shape: " << value.sizes() << ").");

	if (m_last_output && !torch::equal(value, *m_last_output)) {
		std::stringstream ss;
		ss << "Last traced output(shape: " << m_last_output->sizes()
		   << ") does not match given input(shape: " << value.sizes() << ").";
		throw std::runtime_error(ss.str());
	}
}

void InferenceTracer::update_output(torch::Tensor const& value)
{
	static log4cxx::LoggerPtr const logger =
	    log4cxx::Logger::getLogger("hxtorch.InferenceTracer.update_output");
	LOG4CXX_TRACE(logger, "Updating output value(shape: " << value.sizes() << ").");
	m_last_output = value;
}

std::unordered_set<std::shared_ptr<InferenceTracer>>& getInferenceTracer()
{
	static std::unordered_set<std::shared_ptr<InferenceTracer>> storage;
	return storage;
}

void tracer_check_input(torch::Tensor const& value)
{
	for (auto const& tracer : getInferenceTracer()) {
		assert(tracer);
		tracer->check_input(value);
	}
}

void tracer_update_output(torch::Tensor const& value)
{
	for (auto& tracer : getInferenceTracer()) {
		assert(tracer);
		tracer->update_output(value);
	}
}

} // namespace hxtorch::detail
