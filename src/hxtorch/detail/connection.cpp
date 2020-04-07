#include "hxtorch/detail/connection.h"

#include "grenade/vx/config.h"

namespace hxtorch::detail {

std::unique_ptr<hxcomm::vx::ConnectionVariant>& getConnection()
{
	static std::unique_ptr<hxcomm::vx::ConnectionVariant> connection;
	return connection;
}

grenade::vx::ChipConfig& getChip()
{
	static grenade::vx::ChipConfig chip;
	return chip;
}

} // namespace hxtorch::detail
