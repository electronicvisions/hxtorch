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
 * @param hwdb_path Optional path to the hwdb to use
 */
void init(std::optional<std::string> const& hwdb_path = std::nullopt);

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

/**
 * Initialize hardware mock configuration.
 * @param noise_std Standard deviation of noise to add to membrane potentials
 * @param gain Gain to assume for analog multiplication
 */
void init_mock(float noise_std = 2., float gain = 0.0012);

/**
 * Release hardware resource.
 */
void release();

} // namespace hxtorch
