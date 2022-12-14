#include "hxtorch/snn/extract_tensors.h"
#include "grenade/vx/io_data_map.h"
#include "grenade/vx/network/extract_output.h"
#include "grenade/vx/network/network_graph.h"
#include <vector>
#include <torch/torch.h>


// TODO: Use tensor accessors

namespace hxtorch::snn {

std::map<grenade::vx::network::PopulationDescriptor, SpikeHandle> extract_spikes(
    grenade::vx::IODataMap const& data,
    grenade::vx::network::NetworkGraph const& network_graph,
    int runtime)
{
	using namespace grenade::vx;
	using namespace grenade::vx::network;

	// return data
	std::map<PopulationDescriptor, SpikeHandle> ret;

	// return if network has no spike output vertex
	if (!network_graph.get_event_output_vertex()) {
		return ret;
	}

	// spikes
	auto const& spikes = std::get<std::vector<TimedSpikeFromChipSequence>>(
	    data.data.at(*network_graph.get_event_output_vertex()));
	assert(!spikes.size() || spikes.size() == data.batch_size());

	// generate reverse lookup table from spike label to neuron coordinate
	std::map<halco::hicann_dls::vx::SpikeLabel, std::tuple<size_t, PopulationDescriptor>>
	    label_lookup;
	assert(network_graph.get_network());

	// get indices and values of events.
	// NOTE: Would be nicer to use here torch.Tensors right away. However, we do not know the number
	// of events per population trivially beforehand.
	std::map<
	    PopulationDescriptor, std::tuple<std::vector<uint8_t>, std::vector<std::vector<int64_t>>>>
	    indices;

	for (auto const& [descriptor, neurons] : network_graph.get_spike_labels()) {
		if (!std::holds_alternative<Population>(
		        network_graph.get_network()->populations.at(descriptor))) {
			continue;
		}
		auto const& population =
		    std::get<Population>(network_graph.get_network()->populations.at(descriptor));
		for (size_t i = 0; i < neurons.size(); ++i) {
			if (population.enable_record_spikes.at(i)) {
				// internal neurons only have one label assigned
				assert(neurons.at(i).size() == 1);
				assert(neurons.at(i).at(0));
				label_lookup[*(neurons.at(i).at(0))] = std::make_tuple(i, descriptor);
			}
		}
		std::get<1>(indices[descriptor]).resize(3);
	}


	// get data and indices
	for (size_t b = 0; b < spikes.size(); ++b) {
		for (auto const& spike : spikes.at(b)) {
			auto const label = spike.label;
			if (label_lookup.contains(label)) {
				auto const& [in_pop_id, descriptor] = label_lookup.at(label);
				auto& [value, index] = indices[descriptor];
				if (static_cast<int64_t>(spike.chip_time.value()) <= runtime) {
					index.at(0).push_back(static_cast<int64_t>(spike.chip_time.value()));
					index.at(1).push_back(static_cast<int64_t>(b));
					index.at(2).push_back(static_cast<int64_t>(in_pop_id));
					value.push_back(1);
				}
			}
		}
	}

	// create sparse tensors
	for (auto const& [descriptor, pop_data] : indices) {
		auto const& [value, index] = pop_data;

		// tensor options
		auto const& data_options = torch::TensorOptions().dtype(torch::kUInt8);
		auto const& index_options = torch::TensorOptions().dtype(torch::kLong);

		// sparse tensor data
		auto values_tensor = torch::ones({static_cast<int64_t>(value.size())}, data_options);
		auto indicies_tensor = torch::empty({3, static_cast<int64_t>(value.size())}, index_options);

		// convert to tensor
		auto accessor_i = indicies_tensor.accessor<long, 2>();
		for (size_t i = 0; i < value.size(); ++i) {
			for (size_t j = 0; j < 3; ++j) {
				accessor_i[j][i] = index.at(j).at(i);
			}
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
		    1. / 1e6 / static_cast<float>(grenade::vx::TimedSpike::Time::fpga_clock_cycles_per_us));
	}
	return ret;
}


std::map<grenade::vx::network::PopulationDescriptor, MADCHandle> extract_madc(
    grenade::vx::IODataMap const& data,
    grenade::vx::network::NetworkGraph const& network_graph,
    int runtime)
{
	using namespace grenade::vx;
	using namespace grenade::vx::network;

	// return data
	std::map<PopulationDescriptor, MADCHandle> ret;

	if (!network_graph.get_madc_sample_output_vertex()) {
		return ret;
	}

	// get indices and values of events. NOTE: Would be nicer to use here torch.Tensors right away.
	// However, we do not know the number of events per population trivially beforehand.
	std::map<
	    PopulationDescriptor, std::tuple<std::vector<int16_t>, std::vector<std::vector<int64_t>>>>
	    indicies;

	// get indices vectors
	assert(network_graph.get_network()->madc_recording.has_value());
	auto const& recording = network_graph.get_network()->madc_recording.value();
	auto& [value, index] = indicies[recording.population];
	index.resize(3);

	// convert samples
	auto const& samples = std::get<std::vector<TimedMADCSampleFromChipSequence>>(
	    data.data.at(*network_graph.get_madc_sample_output_vertex()));

	assert(!samples.size() || samples.size() == data.batch_size());
	for (size_t b = 0; b < samples.size(); ++b) {
		for (auto const& sample : samples.at(b)) {
			if (static_cast<int64_t>(sample.chip_time.value()) <= runtime) {
				index.at(0).push_back(static_cast<int64_t>(sample.chip_time.value()));
				index.at(1).push_back(static_cast<int64_t>(b));
				index.at(2).push_back(static_cast<int64_t>(recording.index));
				value.push_back(static_cast<int16_t>(sample.value.value()));
			}
		}
	}

	// tensor options
	auto const& data_options = torch::TensorOptions().dtype(torch::kInt16);
	auto const& index_options = torch::TensorOptions().dtype(torch::kLong);

	// tensor data
	auto values_tensor = torch::empty({static_cast<int64_t>(value.size())}, data_options);
	auto indicies_tensor = torch::empty({3, static_cast<int64_t>(value.size())}, index_options);

	// convert to tensor
	auto accessor_v = values_tensor.accessor<int16_t, 1>();
	auto accessor_i = indicies_tensor.accessor<long, 2>();
	for (size_t i = 0; i < value.size(); ++i) {
		for (size_t j = 0; j < 3; ++j) {
			accessor_i[j][i] = index.at(j).at(i);
		}
		accessor_v[i] = value.at(i);
	}

	// create sparse COO tensor
	torch::Tensor madc_tensor = torch::sparse_coo_tensor(
	    indicies_tensor, values_tensor.clone(),
	    {runtime + 1, static_cast<int>(data.batch_size()),
	     static_cast<int>(
	         std::get<Population>(network_graph.get_network()->populations.at(recording.population))
	             .neurons.size())},
	    data_options);

	// handle
	ret[recording.population] = MADCHandle(
	    madc_tensor,
	    1. / 1e6 / static_cast<float>(grenade::vx::TimedSpike::Time::fpga_clock_cycles_per_us));

	return ret;
}


std::map<grenade::vx::network::PopulationDescriptor, CADCHandle> extract_cadc(
    grenade::vx::IODataMap const& data,
    grenade::vx::network::NetworkGraph const& network_graph,
    int runtime)
{
	using namespace grenade::vx;
	using namespace grenade::vx::network;

	// return data
	std::map<PopulationDescriptor, CADCHandle> ret;

	// return if network does not cadc output vertex
	if (!network_graph.get_network()->cadc_recording.has_value()) {
		return ret;
	}

	// get CADC recorded neurons
	auto const& recorded_neurons = network_graph.get_network()->cadc_recording.value().neurons;

	// get indices and values of events. NOTE: Would be nicer to use here torch.Tensors right away.
	// However, we do not know the number of events per population trivially beforehand.
	std::map<
	    PopulationDescriptor, std::tuple<std::vector<int32_t>, std::vector<std::vector<int64_t>>>>
	    indices;

	// create lookup for all neurons recorded
	std::set<PopulationDescriptor> populations;
	std::map<halco::hicann_dls::vx::v3::AtomicNeuronOnDLS, CADCRecording::Neuron> lookup;
	for (auto const& neuron : recorded_neurons) {
		// get population
		auto const& population =
		    std::get<Population>(network_graph.get_network()->populations.at(neuron.population));
		lookup[population.neurons.at(neuron.index)] = neuron;
		populations.insert(neuron.population);
	}

	// resize
	for (auto const& descriptor : populations) {
		std::get<1>(indices[descriptor]).resize(3);
	}

	// get data
	for (auto const cadc_output_vertex : network_graph.get_cadc_sample_output_vertex()) {
		auto const& samples = std::get<std::vector<TimedDataSequence<std::vector<Int8>>>>(
		    data.data.at(cadc_output_vertex));
		assert(!samples.size() || samples.size() == data.batch_size());
		assert(boost::in_degree(cadc_output_vertex, network_graph.get_graph().get_graph()) == 1);
		auto const in_edges =
		    boost::in_edges(cadc_output_vertex, network_graph.get_graph().get_graph());
		auto const cadc_vertex =
		    boost::source(*in_edges.first, network_graph.get_graph().get_graph());
		auto const& vertex = std::get<vertex::CADCMembraneReadoutView>(
		    network_graph.get_graph().get_vertex_property(cadc_vertex));
		auto const& columns = vertex.get_columns();
		auto const& row = vertex.get_synram().toNeuronRowOnDLS();
		for (size_t b = 0; b < samples.size(); ++b) {
			for (auto const& sample : samples.at(b)) {
				for (size_t j = 0; auto const& cs : columns) {
					for (auto const& column : cs) {
						auto const& neuron = lookup[halco::hicann_dls::vx::v3::AtomicNeuronOnDLS(
						    column.toNeuronColumnOnDLS(), row)];
						auto& [value, index] = indices[neuron.population];
						if (static_cast<int64_t>(sample.chip_time.value()) <= runtime) {
							index.at(0).push_back(static_cast<int64_t>(sample.chip_time.value()));
							index.at(1).push_back(static_cast<int64_t>(b));
							index.at(2).push_back(static_cast<int64_t>(neuron.index));
							value.push_back(
							    static_cast<int32_t>(static_cast<int8_t>(sample.data.at(j) + 128)));
						}
						j++;
					}
				}
			}
		}
	}

	for (auto const& [descriptor, pop_data] : indices) {
		auto const& [value, index] = pop_data;

		// tensor options
		auto const& data_options = torch::TensorOptions().dtype(torch::kInt);
		auto const& index_options = torch::TensorOptions().dtype(torch::kLong);

		// sparse tensor data
		auto values_tensor = torch::empty({static_cast<int64_t>(value.size())}, data_options);
		auto indicies_tensor = torch::empty({3, static_cast<int64_t>(value.size())}, index_options);

		// convert to tensor
		auto accessor_v = values_tensor.accessor<int32_t, 1>();
		auto accessor_i = indicies_tensor.accessor<long, 2>();
		for (size_t i = 0; i < value.size(); ++i) {
			for (size_t j = 0; j < 3; ++j) {
				accessor_i[j][i] = index.at(j).at(i);
			}
			accessor_v[i] = value.at(i);
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
		    1. / 1e6 / static_cast<float>(grenade::vx::TimedSpike::Time::fpga_clock_cycles_per_us));
	}
	return ret;
}

} // namespace hxtorch.snn
