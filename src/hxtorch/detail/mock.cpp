#include "hxtorch/detail/mock.h"

namespace hxtorch {

MockParameter& getMockParameter()
{
	static MockParameter mock_parameter{2., 0.0012};
	return mock_parameter;
}

} // namespace hxtorch
