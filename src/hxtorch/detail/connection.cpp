#include "hxtorch/detail/connection.h"

#include "grenade/vx/jit_graph_executor.h"
#include "lola/vx/v3/chip.h"

namespace hxtorch::detail {

std::unique_ptr<grenade::vx::JITGraphExecutor>& getConnection()
{
	static std::unique_ptr<grenade::vx::JITGraphExecutor> connection;
	return connection;
}

lola::vx::v3::Chip& getChip()
{
	static lola::vx::v3::Chip chip;
	return chip;
}

std::unique_ptr<stadls::vx::ReinitStackEntry>& getReinitCalibration()
{
	static std::unique_ptr<stadls::vx::ReinitStackEntry> reinit_calibration;
	return reinit_calibration;
}

} // namespace hxtorch::detail
