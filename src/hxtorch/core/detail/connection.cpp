#include "hxtorch/core/detail/connection.h"

#include "grenade/vx/execution/jit_graph_executor.h"
#include "lola/vx/v3/chip.h"
#include "pyhxcomm/common/handle_connection.h"

namespace hxtorch::core::detail {

std::shared_ptr<pyhxcomm::Handle<grenade::vx::execution::JITGraphExecutor>>& getExecutor()
{
	static std::shared_ptr<pyhxcomm::Handle<grenade::vx::execution::JITGraphExecutor>> executor;
	return executor;
}

lola::vx::v3::Chip& getChip()
{
	static lola::vx::v3::Chip chip;
	return chip;
}

} // namespace hxtorch::core::detail
