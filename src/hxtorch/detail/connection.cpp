#include "hxtorch/detail/connection.h"

#include "grenade/vx/backend/connection.h"
#include "grenade/vx/config.h"

namespace hxtorch::detail {

std::unique_ptr<grenade::vx::backend::Connection>& getConnection()
{
	static std::unique_ptr<grenade::vx::backend::Connection> connection;
	return connection;
}

grenade::vx::ChipConfig& getChip()
{
	static grenade::vx::ChipConfig chip;
	return chip;
}

std::unique_ptr<stadls::vx::ReinitStackEntry>& getReinitCalibration()
{
	static std::unique_ptr<stadls::vx::ReinitStackEntry> reinit_calibration;
	return reinit_calibration;
}

} // namespace hxtorch::detail
