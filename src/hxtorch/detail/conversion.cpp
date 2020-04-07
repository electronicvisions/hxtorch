#include "hxtorch/detail/conversion.h"

namespace hxtorch::detail {

SignedWeight convert_weight(float const value)
{
	SignedWeight ret;
	if (value >= 0) {
		ret.negative = SignedWeight::weight_type(0);
		ret.positive = SignedWeight::weight_type(std::min(value, static_cast<float>(63.)));
	} else {
		ret.positive = SignedWeight::weight_type(0);
		ret.negative = SignedWeight::weight_type(-std::max(value, static_cast<float>(-63.)));
	}
	return ret;
}

grenade::vx::UInt5 convert_activation(float const value)
{
	return grenade::vx::UInt5(std::max(
	    std::min(value, static_cast<float>(grenade::vx::UInt5::max)),
	    static_cast<float>(grenade::vx::UInt5::min)));
}

float convert_membrane(int8_t const value)
{
	return static_cast<float>(value);
}

} // namespace hxtorch::detail
