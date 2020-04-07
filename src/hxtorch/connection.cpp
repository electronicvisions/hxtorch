#include "hxtorch/connection.h"

#include "grenade/vx/config.h"
#include "hxtorch/detail/connection.h"

namespace hxtorch {

void init(
    grenade::vx::ChipConfig const& chip, std::unique_ptr<hxcomm::vx::ConnectionVariant> connection)
{
	detail::getChip() = chip;
	detail::getConnection() = std::move(connection);
}

void release()
{
	detail::getConnection().reset();
}

} // namespace hxtorch
