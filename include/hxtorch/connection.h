#pragma once
#include <optional>
#include <string>

namespace hxtorch {

/**
 * Path to a hardware database.
 */
struct HWDBPath
{
	std::optional<std::string> path;
	std::string version;
	explicit HWDBPath(
	    std::optional<std::string> path = std::nullopt, std::string version = "stable/latest") :
	    path(path), version(version)
	{}
};

/**
 * Initialize the hardware automatically from the environment.
 *
 * @param hwdb_path Optional path to the hwdb to use. Only effective if param `ann` is true.
 * @param ann Bool indicating whether additionally a default chip object is constructed for ANNs
 * from a calibration loaded from `hwdb_path`, or if not given, from the latest nightly calibration.
 */
void init_hardware(std::optional<HWDBPath> const& hwdb_path = std::nullopt, bool ann = false);


/**
 * Path to a calibration.
 */
struct CalibrationPath
{
	std::string value;
	explicit CalibrationPath(std::string value) : value(value) {}
};

/**
 * Initialize the hardware with calibration path.
 *
 * @param calibration_path Calibration path to load from
 */
void init_hardware(CalibrationPath const& calibration_path);

/**
 * Initialize automatically from the environment
 * without ExperimentInit and without any calibration.
 */
void init_hardware_minimal();

/**
 * Get unique identifier
 *
 * @param hwdb_path Optional path to the hwdb to use
 */
std::string get_unique_identifier(std::optional<HWDBPath> const& hwdb_path = std::nullopt);

/**
 * Release hardware resource.
 */
void release_hardware();

} // namespace hxtorch
