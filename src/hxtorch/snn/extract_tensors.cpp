#include "hxtorch/snn/extract_tensors.h"
#include "grenade/vx/network/placed_atomic/network_graph.h"
#include "grenade/vx/network/placed_logical/extract_output.h"
#include "grenade/vx/network/placed_logical/network_graph.h"
#include "grenade/vx/signal_flow/io_data_map.h"
#include <vector>
#include <torch/torch.h>


// TODO: Use tensor accessors

namespace hxtorch::snn {

std::map<grenade::vx::network::placed_logical::PopulationDescriptor, SpikeHandle> extract_spikes(
    grenade::vx::signal_flow::IODataMap const& data,
    grenade::vx::network::placed_logical::NetworkGraph const& network_graph,
    int runtime)
{
	using namespace grenade::vx;
	using namespace grenade::vx::network::placed_logical;

	// return data
	std::map<PopulationDescriptor, SpikeHandle> ret;

	auto const grenade_spikes = extract_neuron_spikes(data, network_graph);

	// get indices of events.
	// NOTE: Would be nicer to use here torch.Tensors right away. However, we do not know the number
	// of events per population trivially beforehand.
	std::map<PopulationDescriptor, std::vector<std::tuple<int64_t, int64_t, int64_t>>> indices;

	assert(network_graph.get_network());
	for (auto const& [descriptor, pop] : network_graph.get_network()->populations) {
		if (!std::holds_alternative<Population>(pop)) {
			continue;
		}
		auto const& neurons = std::get<Population>(pop).neurons;
		if (std::any_of(neurons.begin(), neurons.end(), [](auto const& nrn) {
			    return std::any_of(
			        nrn.compartments.begin(), nrn.compartments.end(), [](auto const& comp) {
				        return comp.second.spike_master &&
				               comp.second.spike_master->enable_record_spikes;
			        });
		    })) {
			indices[descriptor] = {};
		}
	}

	for (size_t b = 0; b < grenade_spikes.size(); ++b) {
		for (auto const& [key, times] : grenade_spikes.at(b)) {
			auto const& [descriptor, neuron_in_population, compartment_in_neuron] = key;
			assert(compartment_in_neuron.value() == 0);
			auto& index = indices[descriptor];
			for (auto const& time : times) {
				if (static_cast<int64_t>(time.value()) <= runtime) {
					index.push_back(std::tuple{
					    static_cast<int64_t>(time.value()), static_cast<int64_t>(b),
					    static_cast<int64_t>(neuron_in_population)});
				}
			}
		}
	}

	for (auto& [_, index] : indices) {
		std::sort(index.begin(), index.end(), [](auto const& a, auto const& b) {
			return std::get<0>(a) < std::get<0>(b);
		});
	}

	// create sparse tensors
	for (auto const& [descriptor, index] : indices) {
		// tensor options
		auto const& data_options = torch::TensorOptions().dtype(torch::kUInt8);
		auto const& index_options = torch::TensorOptions().dtype(torch::kLong);

		// sparse tensor data
		auto values_tensor = torch::ones({static_cast<int64_t>(index.size())}, data_options);
		auto indicies_tensor = torch::empty({3, static_cast<int64_t>(index.size())}, index_options);

		// convert to tensor
		auto accessor_i = indicies_tensor.accessor<long, 2>();
		for (size_t i = 0; i < index.size(); ++i) {
			accessor_i[0][i] = std::get<0>(index.at(i));
			accessor_i[1][i] = std::get<1>(index.at(i));
			accessor_i[2][i] = std::get<2>(index.at(i));
		}

		torch::Tensor spike_tensor = torch::sparse_coo_tensor(
		    indicies_tensor, values_tensor.clone(),
		    {runtime + 1, static_cast<int>(data.batch_size()),
		     static_cast<int>(
		         std::get<Population>(network_graph.get_network()->populations.at(descriptor))
		             .neurons.size())},
		    data_options);

		// handle
		ret[descriptor] = SpikeHandle(
		    spike_tensor,
		    1. / 1e6 / static_cast<float>(grenade::vx::common::Time::fpga_clock_cycles_per_us));
	}
	return ret;
}


std::map<grenade::vx::network::placed_logical::PopulationDescriptor, MADCHandle> extract_madc(
    grenade::vx::signal_flow::IODataMap const& data,
    grenade::vx::network::placed_logical::NetworkGraph const& network_graph,
    int runtime)
{
	using namespace grenade::vx;
	using namespace grenade::vx::network::placed_logical;

	// return data
	std::map<grenade::vx::network::placed_logical::PopulationDescriptor, MADCHandle> ret;

	assert(network_graph.get_network());
	if (!network_graph.get_network()->madc_recording) {
		return ret;
	}

	auto const descriptor = network_graph.get_network()->madc_recording->population;
	auto const neuron_in_population =
	    network_graph.get_network()->madc_recording->neuron_on_population;
	assert(network_graph.get_network()->madc_recording->compartment_on_neuron.value() == 0);

	auto const grenade_samples = extract_madc_samples(data, network_graph);

	std::vector<std::tuple<int16_t, int64_t, int64_t, int64_t>> samples;

	for (size_t b = 0; b < grenade_samples.size(); ++b) {
		for (auto const& [time, value] : grenade_samples.at(b)) {
			samples.push_back(std::tuple{
			    static_cast<int64_t>(value.value()), static_cast<int64_t>(time.value()),
			    static_cast<int64_t>(b), static_cast<int64_t>(neuron_in_population)});
		}
	}
	std::sort(samples.begin(), samples.end(), [](auto const& a, auto const& b) {
		return std::get<1>(a) < std::get<1>(b);
	});

	// tensor options
	auto const& data_options = torch::TensorOptions().dtype(torch::kInt16);
	auto const& index_options = torch::TensorOptions().dtype(torch::kLong);

	// tensor data
	auto values_tensor = torch::empty({static_cast<int64_t>(samples.size())}, data_options);
	auto indicies_tensor = torch::empty({3, static_cast<int64_t>(samples.size())}, index_options);

	// convert to tensor
	auto accessor_v = values_tensor.accessor<int16_t, 1>();
	auto accessor_i = indicies_tensor.accessor<long, 2>();
	for (size_t i = 0; i < samples.size(); ++i) {
		accessor_v[i] = std::get<0>(samples.at(i));
		accessor_i[0][i] = std::get<1>(samples.at(i));
		accessor_i[1][i] = std::get<2>(samples.at(i));
		accessor_i[2][i] = std::get<3>(samples.at(i));
	}

	// create sparse COO tensor
	torch::Tensor madc_tensor = torch::sparse_coo_tensor(
	    indicies_tensor, values_tensor.clone(),
	    {runtime + 1, static_cast<int>(data.batch_size()),
	     static_cast<int>(
	         std::get<Population>(network_graph.get_network()->populations.at(descriptor))
	             .neurons.size())},
	    data_options);

	// handle
	ret[descriptor] = MADCHandle(
	    madc_tensor,
	    1. / 1e6 / static_cast<float>(grenade::vx::common::Time::fpga_clock_cycles_per_us));

	return ret;
}


std::map<grenade::vx::network::placed_logical::PopulationDescriptor, CADCHandle> extract_cadc(
    grenade::vx::signal_flow::IODataMap const& data,
    grenade::vx::network::placed_logical::NetworkGraph const& network_graph,
    int runtime)
{
	using namespace grenade::vx;
	using namespace grenade::vx::network::placed_logical;

	// return data
	std::map<PopulationDescriptor, CADCHandle> ret;

	auto const grenade_samples = extract_cadc_samples(data, network_graph);

	// return if network does not cadc recording
	assert(network_graph.get_network());
	if (!network_graph.get_network()->cadc_recording) {
		return ret;
	}

	// get indices and values of events. NOTE: Would be nicer to use here torch.Tensors right away.
	// However, we do not know the number of events per population trivially beforehand.
	std::map<PopulationDescriptor, std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>>>
	    samples;

	for (size_t b = 0; b < grenade_samples.size(); ++b) {
		for (auto const& [time, population, neuron_on_population, compartment_on_neuron, _, value] :
		     grenade_samples.at(b)) {
			assert(compartment_on_neuron.value() == 0);
			if (static_cast<int64_t>(time.value()) <= runtime) {
				samples[population].push_back(std::tuple{
				    static_cast<int32_t>(static_cast<int8_t>(value + 128)),
				    static_cast<int64_t>(time.value()), static_cast<int64_t>(b),
				    static_cast<int64_t>(neuron_on_population)});
			}
		}
	}
	for (auto& [_, s] : samples) {
		std::sort(s.begin(), s.end(), [](auto const& a, auto const& b) {
			return std::get<1>(a) < std::get<1>(b);
		});
	}

	for (auto const& [descriptor, pop_data] : samples) {
		// tensor options
		auto const& data_options = torch::TensorOptions().dtype(torch::kInt);
		auto const& index_options = torch::TensorOptions().dtype(torch::kLong);

		// sparse tensor data
		auto values_tensor = torch::empty({static_cast<int64_t>(pop_data.size())}, data_options);
		auto indicies_tensor =
		    torch::empty({3, static_cast<int64_t>(pop_data.size())}, index_options);

		// convert to tensor
		auto accessor_v = values_tensor.accessor<int32_t, 1>();
		auto accessor_i = indicies_tensor.accessor<long, 2>();
		for (size_t i = 0; i < pop_data.size(); ++i) {
			accessor_v[i] = std::get<0>(pop_data.at(i));
			accessor_i[0][i] = std::get<1>(pop_data.at(i));
			accessor_i[1][i] = std::get<2>(pop_data.at(i));
			accessor_i[2][i] = std::get<3>(pop_data.at(i));
		}

		torch::Tensor cadc_tensor = torch::sparse_coo_tensor(
		    indicies_tensor, values_tensor.clone(),
		    {runtime + 1, static_cast<int>(data.batch_size()),
		     static_cast<int>(
		         std::get<Population>(network_graph.get_network()->populations.at(descriptor))
		             .neurons.size())},
		    data_options);

		// handle
		ret[descriptor] = CADCHandle(
		    cadc_tensor,
		    1. / 1e6 / static_cast<float>(grenade::vx::common::Time::fpga_clock_cycles_per_us));
	}
	return ret;
}

} // namespace hxtorch.snn
