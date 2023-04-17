#include "hxtorch/core/dense_spikes_to_list.h"

namespace hxtorch::core {

/**
 * Convert dense spike representation to grenade spike representation (vector of vector)
 *
 * { idx: [batch_idx, spike_idx], time: [batch_idx, spike_idx] } -> [batch, neuron_idx, spike_time]
 *
 */
std::vector<std::vector<std::vector<float>>> dense_spikes_to_list(
    std::tuple<pybind11::array_t<int>, pybind11::array_t<float>> spikes, int input_size)
{
	const auto [idx, time] = spikes;
	auto idx_data = idx.unchecked<2>();
	auto time_data = time.unchecked<2>();

	std::vector<std::vector<std::vector<float>>> batches(idx.shape(0));
	for (pybind11::ssize_t batch_idx = 0; batch_idx < idx.shape(0); ++batch_idx) {
		auto& batch = batches[batch_idx];
		batch.resize(input_size); // empty vector for all spikes idx_data

		for (pybind11::ssize_t i = 0; i < idx.shape(1); ++i) {
			const int spike_idx = idx_data(batch_idx, i);
			const float spike_time = time_data(batch_idx, i);
			batch[spike_idx].push_back(spike_time);
		}
	}
	return batches;
}

} // namespace hxtorch::core
