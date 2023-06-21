#pragma once
#include "grenade/vx/network/network_graph.h"
#include "grenade/vx/network/population.h"
#include "hxtorch/spiking/types.h"
#include <map>
#include <torch/torch.h>


namespace grenade::vx {

namespace signal_flow {
class IODataMap;
} // namespace signal_flow

} // namspace grenade::vx


namespace hxtorch::spiking {

/** Convert recorded spikes in IODataMap to population-specific SpikeHandles holding the spikes in a
 * sparse tensor representation.
 *
 * @param data The IODataMap returned by grenade holding all recorded data.
 * @param network_graph The logical grenade graph representation of the network.
 * @param runtime The runtime of the experiment given in FPGA clock cycles.
 * @returns Returns a mapping between population descriptors and spike handles.
 */
std::map<grenade::vx::network::PopulationDescriptor, SpikeHandle> extract_spikes(
    grenade::vx::signal_flow::IODataMap const& data,
    grenade::vx::network::NetworkGraph const& network_graph,
    int runtime);

/** Convert recorded MADC samples in IODataMap to population-specific MADCHandles holding the
 * samples in a sparse tensor representation.
 *
 * @param data The IODataMap returned by grenade holding all recorded data.
 * @param network_graph The logical grenade graph representation of the network.
 * @param runtime The runtime of the experiment given in FPGA clock cycles.
 * @returns Returns a mapping between population descriptors and MADC handles.
 */
std::map<grenade::vx::network::PopulationDescriptor, MADCHandle> extract_madc(
    grenade::vx::signal_flow::IODataMap const& data,
    grenade::vx::network::NetworkGraph const& network_graph,
    int runtime);

/** Convert recorded CADC samples in IODataMap to population-specific CADCHandles holding the
 * samples in a sparse tensor representation.
 *
 * @param data The IODataMap returned by grenade holding all recorded data.
 * @param network_graph The logical grenade graph representation of the network.
 * @param runtime The runtime of the experiment given in FPGA clock cycles.
 * @returns Returns a mapping between population descriptors and CADC handles.
 */
std::map<grenade::vx::network::PopulationDescriptor, CADCHandle> extract_cadc(
    grenade::vx::signal_flow::IODataMap const& data,
    grenade::vx::network::NetworkGraph const& network_graph,
    int runtime);

} // namespace hxtorch::spiking
