#include "hxtorch/core/connection.h"

#include "cereal/types/halco/common/geometry.h"
#include "cereal/types/haldls/cereal.h"
#include "grenade/vx/execution/backend/connection.h"
#include "grenade/vx/execution/backend/run.h"
#include "grenade/vx/execution/jit_graph_executor.h"
#include "hxcomm/vx/connection_from_env.h"
#include "hxtorch/core/detail/connection.h"
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
#include <cereal/types/utility.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <log4cxx/logger.h>

namespace hxtorch::core {

namespace {

lola::vx::v3::Chip const load_and_apply_calibration(std::string calibration_path)
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
	return stadls::vx::v3::convert_to_chip(cocos);
}

} // namespace

void init_hardware_minimal()
{
	detail::getExecutor().reset();
	lola::vx::v3::Chip const chip;
	detail::getChip() = chip;
	grenade::vx::execution::JITGraphExecutor executor;
	detail::getExecutor() =
	    std::make_unique<grenade::vx::execution::JITGraphExecutor>(std::move(executor));
}

void init_hardware(std::optional<HWDBPath> const& hwdb_path, bool ann)
{
	grenade::vx::execution::JITGraphExecutor executor;
	detail::getExecutor() =
	    std::make_unique<grenade::vx::execution::JITGraphExecutor>(std::move(executor));

	if (ann) {
		std::optional<std::string> hwdb_path_value;
		std::string version = "stable/latest";
		if (hwdb_path) {
			hwdb_path_value = hwdb_path->path;
			version = hwdb_path->version;
		}
		using namespace std::string_literals;

		auto const calibration_path = "/wang/data/calibration/hicann-dls-sr-hx/"s +
		                              detail::getExecutor()
		                                  ->get_unique_identifier(hwdb_path_value)
		                                  .at(halco::hicann_dls::vx::DLSGlobal()) +
		                              "/"s + version + "/" + "hagen_cocolist.pbin"s;

		detail::getChip() = load_and_apply_calibration(calibration_path);
	}
}

void init_hardware(CalibrationPath const& calibration_path)
{
	detail::getChip() = load_and_apply_calibration(calibration_path.value);
	grenade::vx::execution::JITGraphExecutor executor;
	detail::getExecutor() =
	    std::make_unique<grenade::vx::execution::JITGraphExecutor>(std::move(executor));
}

std::string get_unique_identifier(std::optional<HWDBPath> const& hwdb_path)
{
	std::optional<std::string> hwdb_path_value;
	if (hwdb_path) {
		hwdb_path_value = hwdb_path->path;
	}
	return detail::getExecutor()
	    ->get_unique_identifier(hwdb_path_value)
	    .at(halco::hicann_dls::vx::DLSGlobal());
}

void release_hardware()
{
	if (detail::getExecutor()) {
		auto connections = detail::getExecutor()->release_connections();
		detail::getExecutor().reset();
		connections.clear();
	}
}

} // namespace hxtorch::core
