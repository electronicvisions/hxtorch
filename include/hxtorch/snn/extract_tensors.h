#pragma once
#include "grenade/vx/network/population.h"
#include "hxtorch/snn/types.h"
#include <map>
#include <torch/torch.h>


namespace grenade::vx {
class IODataMap;

namespace network {
class NetworkGraph;
} // namespace network

} // namspace grenade::vx


namespace hxtorch::snn {

/** Convert recorded spikes in IODataMap to population-specific SpikeHandles holding the spikes in a
 * sparse tensor representation.
 *
 * @param data The IODataMap returned by grenade holding all recorded data.
 * @param network_graph The grenade graph representation of the network.
 * @param runtime The runtime of the experiment given in FPGA clock cycles.
 * @returns Returns a mapping between population descriptors and spike handles.
 */
std::map<grenade::vx::network::PopulationDescriptor, SpikeHandle> extract_spikes(
    grenade::vx::IODataMap const& data,
    grenade::vx::network::NetworkGraph const& network_graph,
    int runtime);

/** Convert recorded MADC samples in IODataMap to population-specific MADCHandles holding the
 * samples in a sparse tensor representation.
 *
 * @param data The IODataMap returned by grenade holding all recorded data.
 * @param network_graph The grenade graph representation of the network.
 * @param runtime The runtime of the experiment given in FPGA clock cycles.
 * @returns Returns a mapping between population descriptors and MADC handles.
 */
std::map<grenade::vx::network::PopulationDescriptor, MADCHandle> extract_madc(
    grenade::vx::IODataMap const& data,
    grenade::vx::network::NetworkGraph const& network_graph,
    int runtime);

/** Convert recorded CADC samples in IODataMap to population-specific CADCHandles holding the
 * samples in a sparse tensor representation.
 *
 * @param data The IODataMap returned by grenade holding all recorded data.
 * @param network_graph The grenade graph representation of the network.
 * @param runtime The runtime of the experiment given in FPGA clock cycles.
 * @returns Returns a mapping between population descriptors and CADC handles.
 */
std::map<grenade::vx::network::PopulationDescriptor, CADCHandle> extract_cadc(
    grenade::vx::IODataMap const& data,
    grenade::vx::network::NetworkGraph const& network_graph,
    int runtime);

} // namespace hxtorch::snn
