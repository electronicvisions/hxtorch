#include "hxtorch/core/extract_data.h"
#include "halco/hicann-dls/vx/v3/event.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace hxtorch::core {

std::tuple<pybind11::array_t<int>, pybind11::array_t<float>> extract_n_spikes(
    std::vector<std::vector<std::vector<grenade::vx::common::Time>>> const& spike_times,
    int n_events,
    int max_spikes)
{
	// create numpy arrays of correct size
	pybind11::array_t<int> numpy_indices(
	    {static_cast<pybind11::ssize_t>(spike_times.size()), // batches
	     static_cast<pybind11::ssize_t>(n_events)});
	pybind11::array_t<float> numpy_values(
	    {static_cast<pybind11::ssize_t>(spike_times.size()), // batches
	     static_cast<pybind11::ssize_t>(n_events)});

	numpy_indices[pybind11::make_tuple(pybind11::ellipsis())] = -1;
	numpy_values[pybind11::make_tuple(pybind11::ellipsis())] =
	    std::numeric_limits<float>::infinity();


	for (size_t b = 0; b < spike_times.size(); ++b) {
		int event_idx = 0;
		for (size_t p = 0; p < spike_times.at(b).size(); ++p) {
			for (size_t t = 0; t < spike_times.at(b).at(p).size(); ++t) {
				auto const time = spike_times.at(b).at(p).at(t).value();
				if (event_idx < std::min(n_events, max_spikes)) {
					numpy_indices.mutable_at(b, event_idx) = p;
					numpy_values.mutable_at(b, event_idx) =
					    time /
					    static_cast<float>(grenade::vx::common::Time::fpga_clock_cycles_per_us);
					event_idx++;
				}
				// FIXME: sort spikes?
			}
		}
	}

	std::tuple<pybind11::array_t<int>, pybind11::array_t<float>> ret =
	    std::make_tuple(numpy_indices, numpy_values);

	return ret;
}


std::tuple<pybind11::array_t<int>, pybind11::array_t<int>> extract_n_madc(
    std::vector<std::vector<std::vector<std::pair<
        grenade::vx::common::Time,
        grenade::vx::signal_flow::MADCSampleFromChip::Value>>>> const& samples,
    int n_samples)
{
	// time stamp
	pybind11::array_t<int> numpy_indices(
	    {static_cast<pybind11::ssize_t>(samples.size()), // batches
	     static_cast<pybind11::ssize_t>(n_samples)});
	// value
	pybind11::array_t<int> numpy_values(
	    {static_cast<pybind11::ssize_t>(samples.size()), // batches
	     static_cast<pybind11::ssize_t>(n_samples)});
	numpy_indices[pybind11::make_tuple(pybind11::ellipsis())] = -1;
	numpy_values[pybind11::make_tuple(pybind11::ellipsis())] = std::numeric_limits<int>::infinity();
	for (size_t b = 0; b < samples.size(); ++b) {
		long unsigned int sample_idx = 0;
		for (size_t p = 0; p < samples.at(b).size(); ++p) {
			for (size_t t = 0; t < samples.at(b).at(p).size(); ++t) {
				auto const& [time, value] = samples.at(b).at(p).at(t);
				if ((sample_idx >= static_cast<size_t>(n_samples)) ||
				    (sample_idx >= samples.at(b).size())) {
					continue;
				}
				numpy_indices.mutable_at(b, sample_idx) = time.value();
				numpy_values.mutable_at(b, sample_idx) = value.value();
				sample_idx++;
			}
		}
	}

	// return data
	std::tuple<pybind11::array_t<int>, pybind11::array_t<int>> ret =
	    std::make_tuple(numpy_indices, numpy_values);
	return ret;
}

} // namespace hxtorch::core
