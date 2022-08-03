#include <hxtorch/snn/detail/to_dense.h>


namespace hxtorch::snn::detail {

torch::Tensor sparse_spike_to_dense(torch::Tensor const& data, float sparse_dt, float dt)
{
	// TODO: At the current implementation we expect a batch dimension. Maybe we should also allow
	// to_dense without a batch dimension.
	assert((sparse_dt > 0) & (dt > 0));
	if (data.dim() != 3) {
		throw std::runtime_error("Only data tensors with dim = 3 are supported.");
	}
	if ((data.sizes()[0] == 0) || (data.sizes()[1] == 0) || (data.sizes()[2] == 0)) {
		throw std::runtime_error("Given data tensor has size = 0 along one dimension.");
	}

	// relative scale
	float scale = dt / sparse_dt;

	// dense tensor
	torch::Tensor dense_spikes = torch::zeros(
	    {data.sizes()[0], static_cast<int>(std::round(data.sizes()[1] / scale) + 1),
	     data.sizes()[2]},
	    torch::TensorOptions().dtype(torch::kUInt8));
	auto accessor_s = dense_spikes.accessor<uint8_t, 3>();

	// accessor for data
	auto const& col_data = data.coalesce();
	auto const& indices = col_data.indices();
	auto accessor_d = indices.accessor<long, 2>();

	// fill values into dense tensor
	for (int val = 0; val < col_data.values().sizes()[0]; ++val) {
		// get time step
		int ts = static_cast<int>(std::round(static_cast<float>(accessor_d[1][val]) / scale));
		// assign spike to dense tensor
		accessor_s[accessor_d[0][val]][ts][accessor_d[2][val]] = 1;
	}
	return dense_spikes;
}


torch::Tensor sparse_cadc_to_dense_linear(torch::Tensor const& data, float sparse_dt, float dt)
{
	if (data.dim() != 3) {
		throw std::runtime_error("Only data tensors with dim = 3 are supported.");
	}
	if ((data.sizes()[0] == 0) || (data.sizes()[1] == 0) || (data.sizes()[2] == 0)) {
		throw std::runtime_error("Given data tensor has size = 0 along one dimension.");
	}
	assert((sparse_dt > 0) & (dt > 0));

	auto const& col_data = data.coalesce();

	// create dense return tensor and fill with interpolated data
	torch::Tensor dense_samples = torch::empty(
	    {col_data.sizes()[0],
	     static_cast<int>(std::round(col_data.sizes()[1] * sparse_dt / dt) + 1),
	     col_data.sizes()[2]},
	    torch::TensorOptions().dtype(torch::kFloat));
	auto a_dense_samples = dense_samples.accessor<float, 3>();

	// sparse data
	auto sparse_values = col_data.values();
	auto sparse_indices = col_data.indices();
	auto a_sparse_values = sparse_values.accessor<int32_t, 1>();
	auto a_sparse_indices = sparse_indices.accessor<long, 2>();

	// assign values
	std::vector<std::vector<float>> running_time_stamps(
	    col_data.sizes()[0], std::vector<float>(col_data.sizes()[2], -1));
	std::vector<std::vector<float>> running_values(
	    col_data.sizes()[0], std::vector<float>(col_data.sizes()[2]));

	// assigne values
	for (int i = 0; i < col_data.values().sizes()[0]; ++i) {
		auto const& b = a_sparse_indices[0][i];
		auto const& ts = a_sparse_indices[1][i];
		auto const& n = a_sparse_indices[2][i];

		// lower bound: index of dense tensor nearest and bigger to previoud time stamp
		int lower_bound = static_cast<int>(
		    std::max((float) (0), std::ceil(running_time_stamps.at(b).at(n) / dt)));
		// upper bound: index of dense tensor nearest and bigger to current time stamp
		int upper_bound = static_cast<int>(std::ceil(static_cast<float>(ts) * sparse_dt / dt));

		// fill
		for (auto t = lower_bound; t < upper_bound; ++t) {
			a_dense_samples[b][t][n] =
			    ((static_cast<float>(a_sparse_values[i]) - running_values.at(b).at(n)) /
			     (ts * sparse_dt - running_time_stamps.at(b).at(n))) *
			        (t * dt - running_time_stamps.at(b).at(n)) +
			    running_values.at(b).at(n);
		}
		// keep value as for lower bound in next round
		running_values.at(b).at(n) = static_cast<float>(a_sparse_values[i]);
		running_time_stamps.at(b).at(n) = ts * sparse_dt;
	}

	// take care of upper ends
	// we pad with the value at the uppermost populated time step
	for (int b = 0; b < col_data.sizes()[0]; ++b) {
		for (int n = 0; n < col_data.sizes()[2]; ++n) {
			// uppermost populated time index
			int upper_index = static_cast<int>(
			    std::ceil(static_cast<float>(running_time_stamps.at(b).at(n)) / dt));
			// fill with last meausred value for this neuron
			for (int t = upper_index; t < dense_samples.sizes()[1]; ++t) {
				a_dense_samples[b][t][n] = running_values.at(b).at(n);
			}
		}
	}

	return dense_samples;
}


torch::Tensor sparse_cadc_to_dense_nn(torch::Tensor const& data, float sparse_dt, float dt)
{
	assert((sparse_dt > 0) & (dt > 0));
	if (data.dim() != 3) {
		throw std::runtime_error("Only data tensors with dim = 3 are supported.");
	}
	if ((data.sizes()[0] == 0) || (data.sizes()[1] == 0) || (data.sizes()[2] == 0)) {
		throw std::runtime_error("Given data tensor has size = 0 along one dimension.");
	}

	auto const& col_data = data.coalesce();

	// create dense return tensor and fill with interpolated data
	torch::Tensor dense_samples = torch::empty(
	    {col_data.sizes()[0],
	     static_cast<int>(std::round(col_data.sizes()[1] * sparse_dt / dt) + 1),
	     col_data.sizes()[2]},
	    torch::TensorOptions().dtype(torch::kFloat));
	auto a_dense_samples = dense_samples.accessor<float, 3>();

	// sparse data
	auto sparse_values = col_data.values();
	auto sparse_indices = col_data.indices();
	auto a_sparse_values = sparse_values.accessor<int32_t, 1>();
	auto a_sparse_indices = sparse_indices.accessor<long, 2>();

	// assign values
	std::vector<std::vector<float>> running_time_stamps(
	    col_data.sizes()[0], std::vector<float>(col_data.sizes()[2], -1));
	std::vector<std::vector<int8_t>> running_values(
	    col_data.sizes()[0], std::vector<int8_t>(col_data.sizes()[2]));

	// assigne values
	for (int i = 0; i < col_data.values().sizes()[0]; ++i) {
		auto const& b = a_sparse_indices[0][i];
		auto const& ts = a_sparse_indices[1][i];
		auto const& n = a_sparse_indices[2][i];

		// lower bound: index of dense tensor nearest and bigger to previoud time stamp
		int lower_bound = static_cast<int>(
		    std::max((float) (0), std::ceil(running_time_stamps.at(b).at(n) / dt)));
		// upper bound: index of dense tensor nearest and bigger to current time stamp
		int upper_bound = static_cast<int>(std::ceil(static_cast<float>(ts) * sparse_dt / dt));

		// fill
		for (auto t = lower_bound; t < upper_bound; ++t) {
			a_dense_samples[b][t][n] = std::abs((t * dt) - running_time_stamps.at(b).at(n)) <
			                                   std::abs((t * dt) - (ts * sparse_dt))
			                               ? running_values.at(b).at(n)
			                               : a_sparse_values[i];
		}
		// keep value as for lower bound in next round
		running_values.at(b).at(n) = a_sparse_values[i];
		running_time_stamps.at(b).at(n) = ts * sparse_dt;
	}

	// take care of upper ends
	// we pad with the value at the uppermost populated time step
	for (int b = 0; b < col_data.sizes()[0]; ++b) {
		for (int n = 0; n < col_data.sizes()[2]; ++n) {
			// uppermost populated time index
			int upper_index = static_cast<int>(
			    std::ceil(static_cast<float>(running_time_stamps.at(b).at(n)) / dt));
			// fill with last meausred value for this neuron
			for (int t = upper_index; t < dense_samples.sizes()[1]; ++t) {
				a_dense_samples[b][t][n] = running_values.at(b).at(n);
			}
		}
	}

	return dense_samples;
}


std::tuple<torch::Tensor, torch::Tensor> sparse_cadc_to_dense_raw(torch::Tensor const& data)
{
	using namespace torch::indexing;

	auto const& col_data = data.coalesce();
	auto sparse_indices = col_data.indices();
	auto sparse_values = col_data.values();

	int min_size = -1;

	std::vector<std::vector<torch::Tensor>> cadc_data;
	std::vector<std::vector<torch::Tensor>> cadc_times;

	for (int b = 0; b < col_data.sizes()[0]; ++b) {
		auto const& b_indices =
		    sparse_indices.index({Slice(), sparse_indices.index({0, Slice()}) == b});
		auto const& b_values = sparse_values.index({sparse_indices.index({0, Slice()}) == b});

		std::vector<torch::Tensor> b_data;
		std::vector<torch::Tensor> b_times;
		for (int n = 0; n < col_data.sizes()[2]; ++n) {
			auto const& n_entry = b_indices.index({Slice(), b_indices.index({2, Slice()}) == n});
			auto const& n_values = b_values.index({b_indices.index({2, Slice()}) == n});
			b_data.push_back(n_values);
			b_times.push_back(n_entry.index({1, Slice()}));
			// update min size
			if (min_size < 0) {
				min_size = n_entry.sizes()[1];
			}
			min_size = n_entry.sizes()[1] < min_size ? n_entry.sizes()[1] : min_size;
		}
		cadc_data.push_back(b_data);
		cadc_times.push_back(b_times);
	}

	torch::Tensor dense_samples = torch::empty(
	    {col_data.sizes()[0], min_size, col_data.sizes()[2]},
	    torch::TensorOptions().dtype(torch::kFloat));
	auto a_dense_samples = dense_samples.accessor<float, 3>();
	torch::Tensor dense_times = torch::empty(
	    {col_data.sizes()[0], min_size, col_data.sizes()[2]},
	    torch::TensorOptions().dtype(torch::kInt));
	auto a_dense_times = dense_times.accessor<int, 3>();

	for (int b = 0; b < col_data.sizes()[0]; ++b) {
		auto const& b_data = cadc_data.at(b);
		auto const& b_times = cadc_times.at(b);
		for (int n = 0; n < col_data.sizes()[2]; ++n) {
			for (int t = 0; t < min_size; ++t) {
				a_dense_samples[b][t][n] = b_data.at(n)[t].item().to<float>();
				a_dense_times[b][t][n] = b_times.at(n)[t].item().to<int>();
			}
		}
	}

	return std::make_tuple(dense_samples, dense_times);
}

} // namespace hxtorch::snn::detail
