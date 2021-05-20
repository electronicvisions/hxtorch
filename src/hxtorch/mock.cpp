#include "hxtorch/mock.h"

#include "hxtorch/detail/mock.h"
#include <stdexcept>

namespace hxtorch {

MockParameter get_mock_parameter()
{
	return hxtorch::detail::getMockParameter();
}

void set_mock_parameter(MockParameter const& parameter)
{
	if ((parameter.gain <= 0) || (parameter.gain > 1)) {
		throw std::overflow_error(
		    "Gain is expected to be in the interval (0, 1] but was " +
		    std::to_string(parameter.gain));
	}
	detail::getMockParameter() = parameter;
}

} // namespace hxtorch
