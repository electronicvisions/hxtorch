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

} // namespace hxtorch::detail
