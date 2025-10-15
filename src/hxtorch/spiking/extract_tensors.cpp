#include "hxtorch/spiking/extract_tensors.h"
#include "grenade/vx/common/time.h"
#include "grenade/vx/signal_flow/event.h"
#include "hate/variant.h"
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace py = pybind11;


namespace hxtorch::spiking {

SpikeHandle extract_spikes(
    std::vector<std::vector<std::vector<grenade::vx::common::Time>>> const& spike_times)
{
	// get indices of events.
	// NOTE: Would be nicer to use here torch.Tensors right away. However, we do not know the number
	// of events per population trivially beforehand.
	std::vector<std::tuple<int64_t, int64_t, int64_t>> indices;

	for (size_t b = 0; b < spike_times.size(); ++b) {
		for (size_t p = 0; p < spike_times.at(b).size(); ++p) {
			for (size_t t = 0; t < spike_times.at(b).at(p).size(); ++t) {
				auto const time = spike_times.at(b).at(p).at(t).value();
				indices.push_back(std::tuple{
				    static_cast<int64_t>(time), static_cast<int64_t>(b), static_cast<int64_t>(p)});
			}
		}
	}

	// create sparse tensors
	SpikeHandle ret = SpikeHandle(
	    std::move(indices), static_cast<int>(spike_times.size()),
	    static_cast<int>(spike_times.at(0).size()));

	return ret;
}


MADCHandle extract_madc(std::vector<std::vector<std::vector<std::pair<
                            grenade::vx::common::Time,
                            grenade::vx::signal_flow::MADCSampleFromChip::Value>>>> const& samples)
{
	std::vector<std::tuple<int16_t, int64_t, int64_t, int64_t>> indices;

	for (size_t b = 0; b < samples.size(); ++b) {
		for (size_t p = 0; p < samples.at(b).size(); ++p) {
			for (size_t t = 0; t < samples.at(b).at(p).size(); ++t) {
				auto const& [time, value] = samples.at(b).at(p).at(t);
				indices.push_back(std::tuple{
				    static_cast<int16_t>(value.value()), static_cast<int64_t>(time.value()),
				    static_cast<int64_t>(b), static_cast<int64_t>(p)});
			}
		}
	}

	// Handle
	MADCHandle ret = MADCHandle(
	    std::move(indices), static_cast<int>(samples.size()),
	    static_cast<int>(samples.at(0).size()));
	return ret;
}


CADCHandle extract_cadc(
    std::vector<std::vector<
        std::vector<std::pair<grenade::vx::common::Time, grenade::vx::signal_flow::Int8>>>> const&
        samples)
{
	std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>> indices;
	for (size_t b = 0; b < samples.size(); ++b) {
		for (size_t p = 0; p < samples.at(b).size(); ++p) {
			for (size_t t = 0; t < samples.at(b).at(p).size(); ++t) {
				auto const& [time, value] = samples.at(b).at(p).at(t);
				indices.push_back(std::tuple{
				    static_cast<int32_t>(static_cast<int8_t>(value.value() + 128)),
				    static_cast<int64_t>(time.value()), static_cast<int64_t>(b),
				    static_cast<int64_t>(p)});
			}
		}
	}
	// Handle
	CADCHandle ret = CADCHandle(
	    std::move(indices), static_cast<int>(samples.size()),
	    static_cast<int>(samples.at(0).size()));

	return ret;
}

} // namespace hxtorch::spiking
