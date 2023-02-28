#include "hxtorch/connection.h"

#include "grenade/vx/execution/backend/connection.h"
#include "grenade/vx/execution/backend/run.h"
#include "grenade/vx/execution/jit_graph_executor.h"
#include "halco/common/cerealization_geometry.h"
#include "hxcomm/vx/connection_from_env.h"
#include "hxtorch/detail/connection.h"
#include "lola/vx/cerealization.h"
#include "lola/vx/v3/chip.h"
#include "stadls/vx/reinit_stack_entry.h"
#include "stadls/vx/v3/dumper.h"
#include "stadls/vx/v3/dumperdone.h"
#include "stadls/vx/v3/init_generator.h"
#include "stadls/vx/v3/playback_program_builder.h"
#include "stadls/vx/v3/run.h"

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

std::tuple<lola::vx::v3::Chip const, stadls::vx::ReinitStackEntry> load_and_apply_calibration(
    std::string calibration_path, grenade::vx::execution::backend::Connection& connection)
{
	auto logger = log4cxx::Logger::getLogger("hxtorch.load_and_apply_calibration");
	LOG4CXX_INFO(logger, "Loading calibration from \"" << calibration_path << "\"");

	stadls::vx::v3::Dumper::done_type cocos;
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
	auto const chip = stadls::vx::v3::convert_to_chip(cocos);

	auto calib_builder = stadls::vx::v3::generate(stadls::vx::v3::ExperimentInit()).builder;
	calib_builder.merge_back(stadls::vx::v3::convert_to_builder(cocos));
	auto calib = calib_builder.done();

	// Register reinit so the calibration gets reappplied whenever we regain control of hw.
	// On direct-access backends, this is a no-op.
	auto reinit_calibration = connection.create_reinit_stack_entry();
	reinit_calibration.set(calib, std::nullopt, true);
	return std::make_tuple(std::move(chip), std::move(reinit_calibration));
}

} // namespace

void init_hardware_minimal()
{
	detail::getConnection().reset();
	auto init_generator = stadls::vx::v3::DigitalInit();
	grenade::vx::execution::backend::Connection connection(
	    hxcomm::vx::get_connection_from_env(), init_generator);
	lola::vx::v3::Chip const chip;
	detail::getChip() = chip;
	std::map<halco::hicann_dls::vx::DLSGlobal, grenade::vx::execution::backend::Connection>
	    connections;
	connections.emplace(halco::hicann_dls::vx::DLSGlobal(), std::move(connection));
	grenade::vx::execution::JITGraphExecutor executor(std::move(connections));
	detail::getConnection() =
	    std::make_unique<grenade::vx::execution::JITGraphExecutor>(std::move(executor));
}


void init_hardware(std::optional<HWDBPath> const& hwdb_path, std::string calib_name)
{
	grenade::vx::execution::backend::Connection connection;

	std::optional<std::string> hwdb_path_value;
	std::string version = "stable/latest";
	if (hwdb_path) {
		hwdb_path_value = hwdb_path->path;
		version = hwdb_path->version;
	}
	using namespace std::string_literals;

	auto const calibration_path = "/wang/data/calibration/hicann-dls-sr-hx/"s +
	                              connection.get_unique_identifier(hwdb_path_value) + "/"s +
	                              version + "/" + calib_name + "_cocolist.pbin"s;

	auto [chip, reinit] = load_and_apply_calibration(calibration_path, connection);
	detail::getChip() = chip;
	std::map<halco::hicann_dls::vx::DLSGlobal, grenade::vx::execution::backend::Connection>
	    connections;
	connections.emplace(halco::hicann_dls::vx::DLSGlobal(), std::move(connection));
	grenade::vx::execution::JITGraphExecutor executor(std::move(connections));
	detail::getConnection() =
	    std::make_unique<grenade::vx::execution::JITGraphExecutor>(std::move(executor));
	detail::getReinitCalibration() =
	    std::make_unique<stadls::vx::ReinitStackEntry>(std::move(reinit));
}

void init_hardware(CalibrationPath const& calibration_path)
{
	grenade::vx::execution::backend::Connection connection;

	auto [chip, reinit] = load_and_apply_calibration(calibration_path.value, connection);
	detail::getChip() = chip;
	std::map<halco::hicann_dls::vx::DLSGlobal, grenade::vx::execution::backend::Connection>
	    connections;
	connections.emplace(halco::hicann_dls::vx::DLSGlobal(), std::move(connection));
	grenade::vx::execution::JITGraphExecutor executor(std::move(connections));
	detail::getConnection() =
	    std::make_unique<grenade::vx::execution::JITGraphExecutor>(std::move(executor));
	detail::getReinitCalibration() =
	    std::make_unique<stadls::vx::ReinitStackEntry>(std::move(reinit));
}


lola::vx::v3::Chip get_chip()
{
	return detail::getChip();
}

void release_hardware()
{
	if (detail::getConnection()) {
		auto connections = detail::getConnection()->release_connections();
		detail::getConnection().reset();
		detail::getReinitCalibration().reset();
		connections.clear();
	}
}

} // namespace hxtorch
