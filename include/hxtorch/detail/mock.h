#pragma once

namespace hxtorch {

struct MockParameter
{
	float noise_std;
	float gain;
};

MockParameter& getMockParameter();

} // namespace hxtorch
