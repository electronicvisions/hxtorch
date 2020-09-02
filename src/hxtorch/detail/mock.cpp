#include "hxtorch/detail/mock.h"

namespace hxtorch::detail {

MockParameter& getMockParameter()
{
	static MockParameter mock_parameter{};
	return mock_parameter;
}

} // namespace hxtorch::detail
