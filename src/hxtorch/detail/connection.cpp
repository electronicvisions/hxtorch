#include "hxtorch/detail/connection.h"

#include "grenade/vx/backend/connection.h"
#include "lola/vx/v2/chip.h"

namespace hxtorch::detail {

std::unique_ptr<grenade::vx::backend::Connection>& getConnection()
{
	static std::unique_ptr<grenade::vx::backend::Connection> connection;
	return connection;
}

lola::vx::v2::Chip& getChip()
{
	static lola::vx::v2::Chip chip;
	return chip;
}

std::unique_ptr<stadls::vx::ReinitStackEntry>& getReinitCalibration()
{
	static std::unique_ptr<stadls::vx::ReinitStackEntry> reinit_calibration;
	return reinit_calibration;
}

} // namespace hxtorch::detail
