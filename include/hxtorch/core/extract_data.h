#pragma once
#include "grenade/vx/network/network_graph.h"
#include "grenade/vx/network/population.h"
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace grenade::vx {

namespace signal_flow {
class IODataMap;
} // namespace signal_flow

} // namspace grenade::vx


namespace hxtorch::core {

/** Convert recorded spikes in IODataMap to population-specific tuples of NumPy arrays holding N
 * spikes for each population in each batch entry. If less spikes are encountered their entry will
 * be np.inf
 *
 * @param data The IODataMap returned by grenade holding all recorded data.
 * @param network_graph The logical grenade graph representation of the network.
 * @param n_spikes The maximal numer of spikes per population.
 * @param runtime The runtime of the experiment given in FPGA clock cycles.
 * @returns Returns a tuple of indices and times, each as numpy array, where the first one holds
 * the neuron index and the second one the spike time corresponding to the index
 */
std::map<
    grenade::vx::network::PopulationOnNetwork,
    std::tuple<pybind11::array_t<int>, pybind11::array_t<float>>>
extract_n_spikes(
    grenade::vx::signal_flow::IODataMap const& data,
    grenade::vx::network::NetworkGraph const& network_graph,
    int runtime,
    std::map<grenade::vx::network::PopulationOnNetwork, int> n_spikes) SYMBOL_VISIBLE;

} // namespace hxtorch::core
