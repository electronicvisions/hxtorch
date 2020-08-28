#include "hxtorch/detail/conversion.h"


namespace hxtorch::detail {

grenade::vx::ComputeSingleMAC::Weight convert_weight(float const value)
{
	return grenade::vx::ComputeSingleMAC::Weight(std::lround(value));
}

grenade::vx::UInt5 convert_activation(float const value)
{
	return grenade::vx::UInt5(std::lround(value));
}

float convert_membrane(int8_t const value)
{
	return static_cast<float>(value);
}

} // namespace hxtorch::detail
