#include "hxtorch/core/extract_data.h"
#include "grenade/vx/network/extract_output.h"
#include "grenade/vx/network/network_graph.h"
#include "grenade/vx/signal_flow/io_data_map.h"
#include "halco/hicann-dls/vx/v3/event.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using namespace grenade::vx::network;

namespace hxtorch::core {

std::map<
    grenade::vx::network::PopulationOnNetwork,
    std::tuple<pybind11::array_t<int>, pybind11::array_t<float>>>
extract_n_spikes(
    grenade::vx::signal_flow::IODataMap const& data,
    grenade::vx::network::NetworkGraph const& network_graph,
    int runtime,
    std::map<grenade::vx::network::PopulationOnNetwork, int> n_spikes)
{
	// return data
	std::map<PopulationOnNetwork, std::tuple<pybind11::array_t<int>, pybind11::array_t<float>>> ret;

	// TODO: SNIP: copied from extra "extract_spikes"
	auto const grenade_spikes = extract_neuron_spikes(data, network_graph);

	// get indices of events.
	std::map<PopulationOnNetwork, std::vector<std::tuple<int64_t, int64_t, int64_t>>> indices;

	assert(network_graph.get_network());

	// create return map with numpy arrays
	for (auto const& [descriptor, population] : network_graph.get_network()->populations) {
		if (!std::holds_alternative<Population>(population)) {
			continue;
		}
		// create numpy arrays of correct size
		pybind11::array_t<int> numpy_indices(
		    {static_cast<pybind11::ssize_t>(grenade_spikes.size()), // batches
		     static_cast<pybind11::ssize_t>(n_spikes[descriptor])});
		pybind11::array_t<float> numpy_values(
		    {static_cast<pybind11::ssize_t>(grenade_spikes.size()), // batches
		     static_cast<pybind11::ssize_t>(n_spikes[descriptor])});
		numpy_indices[pybind11::make_tuple(pybind11::ellipsis())] = -1;
		numpy_values[pybind11::make_tuple(pybind11::ellipsis())] =
		    std::numeric_limits<float>::infinity();
		ret[descriptor] = std::make_tuple(numpy_indices, numpy_values);
	}

	for (size_t b = 0; b < grenade_spikes.size(); ++b) {
		std::map<PopulationOnNetwork, int> event_idx; // new
		for (auto const& [key, times] : grenade_spikes.at(b)) {
			auto const& [descriptor, neuron_in_population, compartment_in_neuron] = key;
			assert(compartment_in_neuron.value() == 0);
			for (auto const& time : times) {
				if (static_cast<int64_t>(time.value()) > runtime)
					continue;
				auto& [indices, values] = ret[descriptor];
				if (event_idx[descriptor] < n_spikes[descriptor]) {
					indices.mutable_at(b, event_idx[descriptor]) = neuron_in_population;
					values.mutable_at(b, event_idx[descriptor]) = time.value();
					event_idx[descriptor]++;
				}
				// FIXME: sort spikes?
			}
		}
	}

	// TODO: SNIP(end): modified from "extract_spikes"

	return ret;
}

} // namespace hxtorch::core
