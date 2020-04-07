#pragma once
#include <memory>
#include "hxcomm/vx/connection_variant.h"

namespace grenade::vx {
class ChipConfig;
} // namespace grenade::vx

namespace hxtorch {

/**
 * Initialize with hardware connection and configuration.
 * @param chip Chip configuration to use
 * @param connection Connection to use
 */
void init(
    grenade::vx::ChipConfig const& chip, std::unique_ptr<hxcomm::vx::ConnectionVariant> connection);

/**
 * Release hardware resource.
 */
void release();

} // namespace hxtorch
