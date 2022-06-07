#pragma once
#include <memory>

namespace grenade::vx::backend {
class Connection;
} // namespace grenade::vx::backend

namespace stadls::vx {
class ReinitStackEntry;
} // namespace stadls::vx

namespace lola::vx::v3 {
class Chip;
} // namespace lola::vx::v3

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
lola::vx::v3::Chip& getChip();

/**
 * Get singleton calibration reinit program.
 * @return Reference to reinit stack entry holding calibration pbmem.
 */
std::unique_ptr<stadls::vx::ReinitStackEntry>& getReinitCalibration();


} // namespace hxtorch::detail
