#pragma once

#include "grenade/vx/compute/mac.h"
#include "grenade/vx/types.h"


namespace hxtorch::detail {

grenade::vx::compute::MAC::Weight convert_weight(float value);

grenade::vx::UInt5 convert_activation(float value);

float convert_membrane(int8_t value);

} // namespace hxtorch::detail
