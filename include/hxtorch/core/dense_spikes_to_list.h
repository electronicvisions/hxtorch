#include "hate/visibility.h"
#include <tuple>
#include <vector>
#include <pybind11/numpy.h>

namespace hxtorch::core {

/**
 * Convert dense spike representation to grenade spike representation
 *
 * { idx: [batch_idx, spike_idx], time: [batch_idx, spike_idx] } -> [batch, neuron_idx, spike_time]
 *
 * @param spikes Pair of NumPy arrays holding the spikes.
 * @return All grenade spikes given as a vector of spikes (batches of neuron indices and times).
 */
std::vector<std::vector<std::vector<float>>> dense_spikes_to_list(
    std::tuple<pybind11::array_t<int>, pybind11::array_t<float>> spikes,
    int input_size) SYMBOL_VISIBLE;

} // namespace hxtorch::core
