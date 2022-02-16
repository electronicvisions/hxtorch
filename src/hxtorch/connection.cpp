#include "hxtorch/connection.h"

#include "grenade/vx/backend/connection.h"
#include "grenade/vx/backend/run.h"
#include "grenade/vx/config.h"
#include "halco/common/cerealization_geometry.h"
#include "hxcomm/vx/connection_from_env.h"
#include "hxtorch/detail/connection.h"
#include "lola/vx/cerealization.h"
#include "stadls/vx/reinit_stack_entry.h"
#include "stadls/vx/v2/dumper.h"
#include "stadls/vx/v2/init_generator.h"
#include "stadls/vx/v2/playback_program_builder.h"
#include "stadls/vx/v2/run.h"

#include <fstream>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
// Needed for manual wrapping (pickling) of Dumper::done_type
#include "hxtorch/detail/mock.h"
#include <cereal/types/utility.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <log4cxx/logger.h>

namespace hxtorch {

namespace {

std::tuple<grenade::vx::ChipConfig const, stadls::vx::ReinitStackEntry> load_and_apply_calibration(
    std::string calibration_path, grenade::vx::backend::Connection& connection)
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
		try {
			cereal::PortableBinaryInputArchive ia(calibration);
			ia(cocos);
		} catch (std::exception const& error) {
			LOG4CXX_ERROR(
			    logger,
			    "Deserializing calibration failed. The deserializer expects portable binary data "
			    "(which typically has a .pbin file extension). Other common errors can be caused "
			    "by mismatch of the hardware abstraction layers used to generate the calibration "
			    "and the ones this library is compiled-against.");
			throw error;
		}
	}
	auto const chip = grenade::vx::convert_to_chip(cocos);

	auto calib_builder = stadls::vx::v2::generate(stadls::vx::v2::ExperimentInit()).builder;
	calib_builder.merge_back(stadls::vx::v2::convert_to_builder(cocos));
	auto calib = calib_builder.done();

	// Register reinit so the calibration gets reappplied whenever we regain control of hw.
	// On direct-access backends, this is a no-op.
	auto reinit_calibration = connection.create_reinit_stack_entry();
	reinit_calibration.set(calib, true);
	return std::make_tuple(std::move(chip), std::move(reinit_calibration));
}

} // namespace

void init_hardware_minimal()
{
	detail::getConnection().reset();
	auto init_generator = stadls::vx::v2::DigitalInit();
	grenade::vx::backend::Connection connection(
	    hxcomm::vx::get_connection_from_env(), init_generator);
	grenade::vx::ChipConfig const chip;
	detail::getChip() = chip;
	detail::getConnection() =
	    std::make_unique<grenade::vx::backend::Connection>(std::move(connection));
}


void init_hardware(std::optional<HWDBPath> const& hwdb_path, bool spiking)
{
	grenade::vx::backend::Connection connection;

	std::optional<std::string> hwdb_path_value;
	std::string version = "stable/latest";
	if (hwdb_path) {
		hwdb_path_value = hwdb_path->path;
		version = hwdb_path->version;
	}
	using namespace std::string_literals;

	auto mode = "hagen";
	if (spiking) {
		mode = "spiking";
	}
	auto const calibration_path = "/wang/data/calibration/hicann-dls-sr-hx/"s +
	                              connection.get_unique_identifier(hwdb_path_value) + "/"s +
	                              version + "/" + mode + "_cocolist.pbin"s;

	auto [chip, reinit] = load_and_apply_calibration(calibration_path, connection);
	detail::getChip() = chip;
	detail::getConnection() =
	    std::make_unique<grenade::vx::backend::Connection>(std::move(connection));
	detail::getReinitCalibration() =
	    std::make_unique<stadls::vx::ReinitStackEntry>(std::move(reinit));
}

void init_hardware(CalibrationPath const& calibration_path)
{
	grenade::vx::backend::Connection connection;

	auto [chip, reinit] = load_and_apply_calibration(calibration_path.value, connection);
	detail::getChip() = chip;
	detail::getConnection() =
	    std::make_unique<grenade::vx::backend::Connection>(std::move(connection));
	detail::getReinitCalibration() =
	    std::make_unique<stadls::vx::ReinitStackEntry>(std::move(reinit));
}


grenade::vx::ChipConfig get_chip()
{
	return detail::getChip();
}

void release_hardware()
{
	detail::getConnection().reset();
}

} // namespace hxtorch
