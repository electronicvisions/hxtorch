#pragma once
#include "lola/vx/v3/chip.h"
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
void init_hardware(std::optional<HWDBPath> const& hwdb_path = std::nullopt, bool spiking = false);


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
 * Get copy of ChipConfig object
 */
lola::vx::v3::Chip get_chip();

/**
 * Release hardware resource.
 */
void release_hardware();

} // namespace hxtorch
