#include "hxtorch/perceptron/detail/mock.h"

namespace hxtorch::perceptron::detail {

MockParameter& getMockParameter()
{
	static MockParameter mock_parameter{};
	return mock_parameter;
}

} // namespace hxtorch::perceptron::detail
