#pragma once
#include "grenade/vx/common/time.h"
#include "grenade/vx/signal_flow/event.h"
#include "grenade/vx/signal_flow/types.h"
#include "hxtorch/spiking/types.h"
#include <map>


namespace hxtorch::spiking {

/** Convert recorded spikes in OutputData to population-specific SpikeHandles holding the spikes in
 * a sparse tensor representation.
 *
 * @param spike_times The spike times
 * @returns SpikeHandles
 */
SpikeHandle extract_spikes(
    std::vector<std::vector<std::vector<grenade::vx::common::Time>>> const& spike_times);

/** Convert recorded MADC samples in OutputData to population-specific MADCHandles holding the
 * samples in a sparse tensor representation.
 *
 * @param samples The MADC samples
 * @returns Returns a mapping between population descriptors and MADC handles.
 */
MADCHandle extract_madc(std::vector<std::vector<std::vector<std::pair<
                            grenade::vx::common::Time,
                            grenade::vx::signal_flow::MADCSampleFromChip::Value>>>> const& samples);

/** Convert recorded CADC samples in OutputData to population-specific CADCHandles holding the
 * samples in a sparse tensor representation.
 *
 * @param data The OutputData returned by grenade holding all recorded data.
 * @returns Returns a mapping between population descriptors and CADC handles.
 */
CADCHandle extract_cadc(
    std::vector<std::vector<
        std::vector<std::pair<grenade::vx::common::Time, grenade::vx::signal_flow::Int8>>>> const&
        samples);

} // namespace hxtorch::spiking
