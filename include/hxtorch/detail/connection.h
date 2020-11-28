#pragma once
#include "hxcomm/vx/connection_variant.h"
#include <memory>

namespace grenade::vx {
class ChipConfig;
} // namespace grenade::vx

namespace hxtorch::detail {

/**
 * Get singleton connection.
 * @return Reference to connection
 */
std::unique_ptr<hxcomm::vx::ConnectionVariant>& getConnection();

/**
 * Get singleton chip configuration.
 * @return Reference to chip configuration
 */
grenade::vx::ChipConfig& getChip();

} // namespace hxtorch::detail
