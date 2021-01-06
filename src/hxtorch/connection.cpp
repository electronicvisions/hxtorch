#include "hxtorch/connection.h"

#include "grenade/vx/config.h"
#include "halco/common/cerealization_geometry.h"
#include "hxcomm/vx/connection_from_env.h"
#include "hxtorch/detail/connection.h"
#include "lola/vx/cerealization.h"
#include "stadls/vx/v2/dumper.h"
#include "stadls/vx/v2/init_generator.h"
#include "stadls/vx/v2/playback_program_builder.h"
#include "stadls/vx/v2/run.h"

#include <fstream>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
// Needed for manual wrapping (pickling) of Dumper::done_type
#include "hxtorch/detail/mock.h"
#include <cereal/types/utility.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <log4cxx/logger.h>

namespace hxtorch {

namespace {

grenade::vx::ChipConfig load_and_apply_calibration(
    std::string calibration_path, hxcomm::vx::ConnectionVariant& connection)
{
	auto logger = log4cxx::Logger::getLogger("hxtorch.load_and_apply_calibration");
	LOG4CXX_INFO(logger, "Loading calibration from \"" << calibration_path << "\"");

	stadls::vx::v2::Dumper::done_type cocos;
	{
		std::ifstream calibration(calibration_path);
		if (!calibration.is_open()) {
			throw std::runtime_error(
			    std::string("Failed to open calibration at ") + calibration_path + ".");
		}
		{
			cereal::BinaryInputArchive ia(calibration);
			ia(cocos);
		}
	}
	auto const chip = grenade::vx::convert_to_chip(cocos);
	stadls::vx::v2::run(connection, stadls::vx::v2::convert_to_builder(cocos).done());
	return chip;
}

} // namespace

void init_hardware_minimal()
{
	detail::getConnection().reset();
	auto connection = hxcomm::vx::get_connection_from_env();
	auto init_generator = stadls::vx::v2::DigitalInit();
	stadls::vx::v2::run(connection, stadls::vx::v2::generate(init_generator).builder.done());
	grenade::vx::ChipConfig const chip;
	detail::getChip() = chip;
	detail::getConnection() =
	    std::make_unique<hxcomm::vx::ConnectionVariant>(std::move(connection));
}


void init_hardware(std::optional<HWDBPath> const& hwdb_path)
{
	auto connection = hxcomm::vx::get_connection_from_env();

	std::optional<std::string> hwdb_path_value;
	std::string version = "stable/latest";
	if (hwdb_path) {
		hwdb_path_value = hwdb_path->path;
		version = hwdb_path->version;
	}
	using namespace std::string_literals;
	auto const calibration_path =
	    "/wang/data/calibration/hicann-dls-sr-hx/"s +
	    std::visit(
	        [hwdb_path_value](auto const& c) { return c.get_unique_identifier(hwdb_path_value); },
	        connection) +
	    "/"s + version + "/hagen_cocolist.bin"s;

	stadls::vx::v2::run(
	    connection, stadls::vx::v2::generate(stadls::vx::v2::ExperimentInit()).builder.done());

	auto const chip = load_and_apply_calibration(calibration_path, connection);
	detail::getChip() = chip;
	detail::getConnection() =
	    std::make_unique<hxcomm::vx::ConnectionVariant>(std::move(connection));
}

void init_hardware(CalibrationPath const& calibration_path)
{
	auto connection = hxcomm::vx::get_connection_from_env();
	stadls::vx::v2::run(
	    connection, stadls::vx::v2::generate(stadls::vx::v2::ExperimentInit()).builder.done());

	auto const chip = load_and_apply_calibration(calibration_path.value, connection);
	detail::getChip() = chip;
	detail::getConnection() =
	    std::make_unique<hxcomm::vx::ConnectionVariant>(std::move(connection));
}

void release_hardware()
{
	detail::getConnection().reset();
}

} // namespace hxtorch
