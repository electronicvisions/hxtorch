#include "hxtorch/core/detail/connection.h"

#include "grenade/vx/execution/jit_graph_executor.h"
#include "lola/vx/v3/chip.h"

namespace hxtorch::core::detail {

std::unique_ptr<grenade::vx::execution::JITGraphExecutor>& getExecutor()
{
	static std::unique_ptr<grenade::vx::execution::JITGraphExecutor> executor;
	return executor;
}

lola::vx::v3::Chip& getChip()
{
	static lola::vx::v3::Chip chip;
	return chip;
}

} // namespace hxtorch::core::detail
