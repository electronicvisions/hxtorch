#pragma once
#include <memory>
#include <optional>
#include <string>
#include "hxcomm/vx/connection_variant.h"

namespace grenade::vx {
class ChipConfig;
} // namespace grenade::vx

namespace hxtorch {

/**
 * Initialize automatically from the environment.
 * @param calibration_version Calibration version to load
 * @param hwdb_path Optional path to the hwdb to use
 */
void init(
    std::string calibration_version = "stable/latest",
    std::optional<std::string> const& hwdb_path = std::nullopt);

/**
 * Initialize with hardware connection and calibration path.
 * @param calibration_path Calibration path to load from
 * @param connection Connection to use
 */
void init(
    std::string const& calibration_path, std::unique_ptr<hxcomm::vx::ConnectionVariant> connection);

/**
 * Initialize with hardware connection and configuration.
 * @param chip Chip configuration to use
 * @param connection Connection to use
 */
void init(
    grenade::vx::ChipConfig const& chip, std::unique_ptr<hxcomm::vx::ConnectionVariant> connection);

struct MockParameter;

/**
 * Initialize hardware mock configuration.
 * @param parameter Parameter to use
 */
void init(MockParameter const& parameter);

/**
 * Release hardware resource.
 */
void release();

} // namespace hxtorch
