#pragma once
#include <memory>

namespace grenade::vx {
class ChipConfig;
namespace backend {
class Connection;
} // namespace backend
} // namespace grenade::vx

namespace stadls::vx {
class ReinitStackEntry;
} // namespace stadls::vx

namespace hxtorch::detail {

/**
 * Get singleton connection.
 * @return Reference to connection
 */
std::unique_ptr<grenade::vx::backend::Connection>& getConnection();

/**
 * Get singleton chip configuration.
 * @return Reference to chip configuration
 */
grenade::vx::ChipConfig& getChip();

/**
 * Get singleton calibration reinit program.
 * @return Reference to reinit stack entry holding calibration pbmem.
 */
std::unique_ptr<stadls::vx::ReinitStackEntry>& getReinitCalibration();


} // namespace hxtorch::detail
