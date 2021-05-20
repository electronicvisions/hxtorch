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
 * @param calibration_version Calibration version to load
 * @param hwdb_path Optional path to the hwdb to use
 */
void init(std::optional<HWDBPath> const& hwdb_path = std::nullopt);

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
void init(CalibrationPath const& calibration_path);

/**
 * Release hardware resource.
 */
void release();

} // namespace hxtorch
