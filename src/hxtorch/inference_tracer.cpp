#include "hxtorch/inference_tracer.h"

#include "hxtorch/detail/inference_tracer.h"

namespace hxtorch {

InferenceTracer::InferenceTracer(std::string const& filename) : m_filename(filename), m_impl() {}

void InferenceTracer::start()
{
	m_impl = std::make_shared<detail::InferenceTracer>();
	detail::getInferenceTracer().insert(m_impl);
}

void InferenceTracer::stop()
{
	assert(m_impl);
	detail::getInferenceTracer().erase(m_impl);

	// currently only save all operation names
	{
		std::fstream file(m_filename);
		for (auto const& op : m_impl->operation_names) {
			file << op << std::endl;
		}
	}

	m_impl.reset();
}

} // namespace hxtorch
