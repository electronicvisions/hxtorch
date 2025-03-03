#pragma once
#include "pyhxcomm/common/handle_connection.h"
#include <memory>

namespace grenade::vx::execution {
class JITGraphExecutor;
} // namespace grenade::vx::execution

namespace lola::vx::v3 {
class Chip;
} // namespace lola::vx::v3

namespace hxtorch::core::detail {

/**
 * Get singleton executor.
 * @return Reference to executor
 */
std::shared_ptr<pyhxcomm::Handle<grenade::vx::execution::JITGraphExecutor>>& getExecutor();

/**
 * Get singleton chip configuration.
 * @return Reference to chip configuration
 */
lola::vx::v3::Chip& getChip();

} // namespace hxtorch::core::detail
