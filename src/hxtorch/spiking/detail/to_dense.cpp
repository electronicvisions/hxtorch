#include "grenade/vx/common/time.h"
#include <hxtorch/spiking/detail/to_dense.h>


namespace hxtorch::spiking::detail {


torch::Tensor sparse_spike_to_dense(
    std::vector<std::tuple<int64_t, int64_t, int64_t>> const& data,
    int batch_size,
    int population_size,
    float runtime,
    float dt)
{
	// TODO: At the current implementation we expect a batch dimension. Maybe we should also allow
	// to_dense without a batch dimension.
	assert((runtime > 0) & (dt > 0));

	// dense tensor
	torch::Tensor spikes = torch::zeros(
	    {static_cast<int>(std::ceil(runtime / dt) + 1), batch_size, population_size},
	    torch::TensorOptions().dtype(torch::kUInt8));
	auto a_spikes = spikes.accessor<uint8_t, 3>();

	// fill values into dense tensor
	for (auto const& [time, b, n] : data) {
		int t = static_cast<int>(std::round(
		    static_cast<float>(time) /
		    static_cast<float>(grenade::vx::common::Time::fpga_clock_cycles_per_us) / 1e6 / dt));
		if (t < spikes.sizes()[0]) {
			a_spikes[t][b][n] = 1;
		};
	}

	return spikes;
}


torch::Tensor sparse_madc_to_dense_raw(
    std::vector<std::tuple<int16_t, int64_t, int64_t, int64_t>> const& data, int batch_size)
{
	// time steps
	int time_steps = data.size() / batch_size;

	torch::Tensor samples =
	    torch::empty({2, time_steps, batch_size}, torch::TensorOptions().dtype(torch::kFloat));
	auto a_samples = samples.accessor<float, 3>();

	std::vector<int> running_index(batch_size, 0);
	for (auto const& [value, time, b, _] : data) {
		if (running_index.at(b) < time_steps) {
			a_samples[1][running_index.at(b)][b] = value;
			a_samples[0][running_index.at(b)][b] = time;
			running_index.at(b)++;
		}
	}

	return samples;
}


torch::Tensor sparse_cadc_to_dense_linear(
    std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>> const& data,
    int batch_size,
    int population_size,
    float runtime,
    float dt)
{
	assert((runtime > 0) & (dt > 0));

	// create dense return tensor and fill with interpolated data
	torch::Tensor samples = torch::empty(
	    {static_cast<int>(std::ceil(runtime / dt) + 1), batch_size, population_size},
	    torch::TensorOptions().dtype(torch::kFloat));
	auto a_samples = samples.accessor<float, 3>();

	// assign values
	std::vector<std::vector<float>> running_time_stamps(
	    batch_size, std::vector<float>(population_size, -1));
	std::vector<std::vector<float>> running_values(batch_size, std::vector<float>(population_size));

	// assigne values
	for (auto const& [value, time, b, n] : data) {
		// lower bound: index of dense tensor nearest and bigger to previous time stamp
		int lower_bound =
		    static_cast<int>(std::max(0.f, std::ceil(running_time_stamps.at(b).at(n) / dt)));
		// upper bound: index of dense tensor nearest and bigger to current time stamp
		int upper_bound = static_cast<int>(std::ceil(
		    static_cast<float>(time) /
		    static_cast<float>(grenade::vx::common::Time::fpga_clock_cycles_per_us) / 1e6 / dt));

		// fill
		for (auto t = lower_bound; t < upper_bound; ++t) {
			if (t < samples.sizes()[0]) {
				a_samples[t][b][n] =
				    ((static_cast<float>(value) - running_values.at(b).at(n)) /
				     (static_cast<float>(time) /
				          static_cast<float>(grenade::vx::common::Time::fpga_clock_cycles_per_us) /
				          1e6 -
				      running_time_stamps.at(b).at(n))) *
				        (t * dt - running_time_stamps.at(b).at(n)) +
				    running_values.at(b).at(n);
			}
		}
		// keep value as for lower bound in next round
		running_values.at(b).at(n) = static_cast<float>(value);
		running_time_stamps.at(b).at(n) =
		    static_cast<float>(time) /
		    static_cast<float>(grenade::vx::common::Time::fpga_clock_cycles_per_us) / 1e6;
	}

	// TODO: Make this more efficient
	// take care of upper ends
	// we pad with the value at the uppermost populated time step
	for (int b = 0; b < batch_size; ++b) {
		for (int n = 0; n < population_size; ++n) {
			// uppermost populated time index
			int upper_index = static_cast<int>(
			    std::ceil(static_cast<float>(running_time_stamps.at(b).at(n)) / dt));
			// fill with last measured value for this neuron
			for (int t = upper_index; t < samples.sizes()[0]; ++t) {
				a_samples[t][b][n] = running_values.at(b).at(n);
			}
		}
	}

	return samples;
}


torch::Tensor sparse_cadc_to_dense_nn(
    std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>> const& data,
    int batch_size,
    int population_size,
    float runtime,
    float dt)
{
	assert((runtime > 0) & (dt > 0));

	// create dense return tensor and fill with interpolated data
	torch::Tensor samples = torch::empty(
	    {static_cast<int>(std::ceil(runtime / dt) + 1), batch_size, population_size},
	    torch::TensorOptions().dtype(torch::kFloat));
	auto a_samples = samples.accessor<float, 3>();

	// assign values
	std::vector<std::vector<float>> running_time_stamps(
	    batch_size, std::vector<float>(population_size, -1));
	std::vector<std::vector<float>> running_values(batch_size, std::vector<float>(population_size));

	// assign values
	for (auto const& [value, time, b, n] : data) {
		// lower bound: index of dense tensor nearest and bigger to previous time stamp
		int lower_bound =
		    static_cast<int>(std::max(0.f, std::ceil(running_time_stamps.at(b).at(n) / dt)));
		// upper bound: index of dense tensor nearest and bigger to current time stamp
		int upper_bound = static_cast<int>(std::ceil(
		    static_cast<float>(time) /
		    static_cast<float>(grenade::vx::common::Time::fpga_clock_cycles_per_us) / 1e6 / dt));

		// fill
		for (auto t = lower_bound; t < upper_bound; ++t) {
			if (t < samples.sizes()[0]) {
				a_samples[t][b][n] =
				    std::abs((t * dt) - running_time_stamps.at(b).at(n)) <
				            std::abs(
				                (t * dt) -
				                (time /
				                 static_cast<float>(
				                     grenade::vx::common::Time::fpga_clock_cycles_per_us) /
				                 1e6))
				        ? running_values.at(b).at(n)
				        : value;
			}
		}
		// keep value as for lower bound in next round
		running_values.at(b).at(n) = static_cast<float>(value);
		running_time_stamps.at(b).at(n) =
		    time / static_cast<float>(grenade::vx::common::Time::fpga_clock_cycles_per_us) / 1e6;
	}

	// TODO: Make this more efficient
	// take care of upper ends
	// we pad with the value at the uppermost populated time step
	for (int b = 0; b < batch_size; ++b) {
		for (int n = 0; n < population_size; ++n) {
			// uppermost populated time index
			int upper_index = static_cast<int>(
			    std::ceil(static_cast<float>(running_time_stamps.at(b).at(n)) / dt));
			// fill with last measured value for this neuron
			for (int t = upper_index; t < samples.sizes()[0]; ++t) {
				a_samples[t][b][n] = running_values.at(b).at(n);
			}
		}
	}

	return samples;
}


std::tuple<torch::Tensor, torch::Tensor> sparse_cadc_to_dense_raw(
    std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>> const& data,
    int batch_size,
    int population_size)
{
	// get minimum number of time steps per batch and neuron
	std::vector<int> count_time_steps(batch_size * population_size, 0);
	for (auto const& [value, time, b, n] : data) {
		++count_time_steps.at(b * population_size + n);
	}
	int time_steps = *std::min_element(count_time_steps.begin(), count_time_steps.end());

	torch::Tensor samples = torch::empty(
	    {time_steps, batch_size, population_size}, torch::TensorOptions().dtype(torch::kFloat));
	auto a_samples = samples.accessor<float, 3>();

	torch::Tensor times = torch::empty(
	    {time_steps, batch_size, population_size}, torch::TensorOptions().dtype(torch::kInt));
	auto a_times = times.accessor<int, 3>();

	std::vector<std::vector<int>> running_index(batch_size, std::vector<int>(population_size, 0));
	for (auto const& [value, time, b, n] : data) {
		if (running_index.at(b).at(n) < time_steps) {
			a_samples[running_index.at(b).at(n)][b][n] = value;
			a_times[running_index.at(b).at(n)][b][n] = time;
			running_index.at(b).at(n)++;
		}
	}

	return std::make_tuple(samples, times);
}

} // namespace hxtorch::spiking::detail
