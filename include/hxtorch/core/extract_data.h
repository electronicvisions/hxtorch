#pragma once
#include "grenade/vx/common/time.h"
#include "grenade/vx/signal_flow/event.h"
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>


namespace hxtorch::core {

/** Convert recorded spikes in OutputData to population-specific tuples of NumPy arrays holding N
 * spikes for each population in each batch entry. If less spikes are encountered their entry will
 * be np.inf
 *
 * @param spike_times The spike times per batch, population and neuron.
 * @param n_spikes The maximal numer of spikes per population.
 * @returns Returns a tuple of indices and times, each as numpy array, where the first one holds
 * the neuron index and the second one the spike time corresponding to the index
 */
std::tuple<pybind11::array_t<int>, pybind11::array_t<float>> extract_n_spikes(
    std::vector<std::vector<std::vector<grenade::vx::common::Time>>> const& spike_times,
    int n_events,
    int max_spikes) SYMBOL_VISIBLE;

std::tuple<pybind11::array_t<int>, pybind11::array_t<int>> extract_n_madc(
    std::vector<std::vector<std::vector<std::pair<
        grenade::vx::common::Time,
        grenade::vx::signal_flow::MADCSampleFromChip::Value>>>> const& samples,
    int n_samples) SYMBOL_VISIBLE;

} // namespace hxtorch::core
