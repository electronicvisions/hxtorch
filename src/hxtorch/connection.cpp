#include "hxtorch/connection.h"

#include "grenade/vx/config.h"
#include "halco/common/cerealization_geometry.h"
#include "hxcomm/vx/connection_from_env.h"
#include "hxtorch/detail/connection.h"
#include "lola/vx/cerealization.h"
#include "stadls/vx/dumper.h"
#include "stadls/vx/init_generator.h"
#include "stadls/vx/playback_program_builder.h"
#include "stadls/vx/run.h"

#include <fstream>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
// Needed for manual wrapping (pickling) of Dumper::done_type
#include <cereal/types/utility.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>

namespace hxtorch {

void init(std::optional<std::string> const& hwdb_path)
{
	auto connection = hxcomm::vx::get_connection_from_env();

	using namespace std::string_literals;
	auto const calibration_path =
	    "/wang/data/calibration/hicann-dls-sr-hx/"s +
	    std::visit(
	        [hwdb_path](auto const& c) { return c.get_unique_identifier(hwdb_path); }, connection) +
	    "/stable/latest/hagen_cocolist.bin"s;

	stadls::vx::run(connection, stadls::vx::generate(stadls::vx::ExperimentInit()).builder.done());

	init(calibration_path, std::make_unique<hxcomm::vx::ConnectionVariant>(std::move(connection)));
}

void init(
    std::string const& calibration_path, std::unique_ptr<hxcomm::vx::ConnectionVariant> connection)
{
	stadls::vx::Dumper::done_type cocos;
	{
		std::ifstream calibration(calibration_path);
		{
			cereal::BinaryInputArchive ia(calibration);
			ia(cocos);
		}
	}
	auto const chip = grenade::vx::convert_to_chip(cocos);

	// apply calibration
	if (!connection) {
		throw std::runtime_error("No connection allocated.");
	}
	stadls::vx::run(*connection, stadls::vx::convert_to_builder(cocos).done());

	init(chip, std::move(connection));
}

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
