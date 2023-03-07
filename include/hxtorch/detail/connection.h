#pragma once
#include <memory>

namespace grenade::vx::execution {
class JITGraphExecutor;
} // namespace grenade::vx::execution

namespace lola::vx::v3 {
class Chip;
} // namespace lola::vx::v3

namespace hxtorch::detail {

/**
 * Get singleton executor.
 * @return Reference to executor
 */
std::unique_ptr<grenade::vx::execution::JITGraphExecutor>& getExecutor();

/**
 * Get singleton chip configuration.
 * @return Reference to chip configuration
 */
lola::vx::v3::Chip& getChip();

} // namespace hxtorch::detail
