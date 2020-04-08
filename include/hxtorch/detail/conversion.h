#pragma once

#include "grenade/vx/compute_single_mac.h"
#include "grenade/vx/types.h"


namespace hxtorch::detail {

struct SignedWeight
{
	typedef grenade::vx::ComputeSingleMAC::Weights::value_type::value_type weight_type;
	weight_type positive;
	weight_type negative;
};

SignedWeight convert_weight(float value);

grenade::vx::UInt5 convert_activation(float value);

float convert_membrane(int8_t value);

} // namespace hxtorch::detail
