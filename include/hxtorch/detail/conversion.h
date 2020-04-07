#pragma once

#include "grenade/vx/types.h"
#include "haldls/vx/synapse.h"

namespace hxtorch::detail {

struct SignedWeight
{
	typedef haldls::vx::SynapseQuad::Weight weight_type;
	weight_type positive;
	weight_type negative;
};

SignedWeight convert_weight(float value);

grenade::vx::UInt5 convert_activation(float value);

float convert_membrane(int8_t value);

} // namespace hxtorch::detail
