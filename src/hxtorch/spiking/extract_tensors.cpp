#include "hxtorch/spiking/extract_tensors.h"
#include "grenade/vx/network/extract_output.h"
#include "grenade/vx/network/network_graph.h"
#include "grenade/vx/signal_flow/output_data.h"
#include "hate/variant.h"
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

namespace py = pybind11;


// TODO: Use tensor accessors

namespace hxtorch::spiking {

std::map<grenade::vx::network::PopulationOnNetwork, SpikeHandle> extract_spikes(
    grenade::vx::signal_flow::OutputData const& data,
    grenade::vx::network::NetworkGraph const& network_graph)
{
	using namespace grenade::vx;
	using namespace grenade::vx::network;

	// return data
	std::map<PopulationOnNetwork, SpikeHandle> ret;

	auto const& grenade_spikes = extract_neuron_spikes(data, network_graph);

	// get indices of events.
	// NOTE: Would be nicer to use here torch.Tensors right away. However, we do not know the number
	// of events per population trivially beforehand.
	std::map<PopulationOnNetwork, std::vector<std::tuple<int64_t, int64_t, int64_t>>> indices;

	assert(network_graph.get_network());
	for (auto const& [id, execution_instance] : network_graph.get_network()->execution_instances) {
		for (auto const& [descriptor, pop] : execution_instance.populations) {
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
				indices[PopulationOnNetwork(descriptor, id)] = {};
			}
		}
	}

	for (size_t b = 0; b < grenade_spikes.size(); ++b) {
		for (auto const& [key, times] : grenade_spikes.at(b)) {
			auto const& [descriptor, neuron_in_population, compartment_in_neuron] = key;
			assert(compartment_in_neuron.value() == 0);
			auto& index = indices[descriptor];
			for (auto const& time : times) {
				index.push_back(std::tuple{
				    static_cast<int64_t>(time.value()), static_cast<int64_t>(b),
				    static_cast<int64_t>(neuron_in_population)});
			}
		}
	}

	// create sparse tensors
	for (auto const& [descriptor, index] : indices) {
		ret[descriptor] = SpikeHandle(
		    std::move(index), static_cast<int>(data.batch_size()),
		    static_cast<int>(std::visit(
		        [](auto const& pop) { return pop.neurons.size(); },
		        network_graph.get_network()
		            ->execution_instances.at(descriptor.toExecutionInstanceID())
		            .populations.at(descriptor.toPopulationOnExecutionInstance()))));
	}

	return ret;
}


std::map<grenade::vx::network::PopulationOnNetwork, MADCHandle> extract_madc(
    grenade::vx::signal_flow::OutputData const& data,
    grenade::vx::network::NetworkGraph const& network_graph)
{
	using namespace grenade::vx;
	using namespace grenade::vx::network;

	// return data
	std::map<grenade::vx::network::PopulationOnNetwork, MADCHandle> ret;

	auto const& grenade_samples = extract_madc_samples(data, network_graph);

	assert(network_graph.get_network());
	for (auto const& [id, execution_instance] : network_graph.get_network()->execution_instances) {
		if (!execution_instance.madc_recording) {
			continue;
		}

		// TODO: support two channels
		if (execution_instance.madc_recording->neurons.size() != 1) {
			throw std::runtime_error("Unsupported number of recorded MADC channels.");
		}
		auto const descriptor =
		    execution_instance.madc_recording->neurons.at(0).coordinate.population;
		auto const neuron_in_population =
		    execution_instance.madc_recording->neurons.at(0).coordinate.neuron_on_population;
		assert(
		    execution_instance.madc_recording->neurons.at(0)
		        .coordinate.compartment_on_neuron.value() == 0);

		std::vector<std::tuple<int16_t, int64_t, int64_t, int64_t>> samples;

		for (size_t b = 0; b < grenade_samples.size(); ++b) {
			for (auto const& [time, atomic_neuron_on_network, value] : grenade_samples.at(b)) {
				if (atomic_neuron_on_network.population.toExecutionInstanceID() != id) {
					continue;
				}
				samples.push_back(std::tuple{
				    static_cast<int64_t>(value.value()), static_cast<int64_t>(time.value()),
				    static_cast<int64_t>(b), static_cast<int64_t>(neuron_in_population)});
			}
		}

		// handle
		ret[PopulationOnNetwork(descriptor, id)] = MADCHandle(
		    std::move(samples), static_cast<int>(data.batch_size()),
		    static_cast<int>(std::get<Population>(execution_instance.populations.at(descriptor))
		                         .neurons.size()));
	}
	return ret;
}


std::map<grenade::vx::network::PopulationOnNetwork, CADCHandle> extract_cadc(
    grenade::vx::signal_flow::OutputData const& data,
    grenade::vx::network::NetworkGraph const& network_graph)
{
	using namespace grenade::vx;
	using namespace grenade::vx::network;

	// return data
	std::map<PopulationOnNetwork, CADCHandle> ret;

	auto const& grenade_samples = extract_cadc_samples(data, network_graph);

	// get indices and values of events. NOTE: Would be nicer to use here torch.Tensors right away.
	// However, we do not know the number of events per population trivially beforehand.
	std::map<PopulationOnNetwork, std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>>>
	    samples;

	assert(network_graph.get_network());
	for (auto const& [id, execution_instance] : network_graph.get_network()->execution_instances) {
		if (!execution_instance.cadc_recording) {
			continue;
		}

		for (size_t b = 0; b < grenade_samples.size(); ++b) {
			for (auto const& [time, atomic_neuron_on_network, value] : grenade_samples.at(b)) {
				if (atomic_neuron_on_network.population.toExecutionInstanceID() != id) {
					continue;
				}
				assert(atomic_neuron_on_network.compartment_on_neuron.value() == 0);
				samples[atomic_neuron_on_network.population].push_back(std::tuple{
				    static_cast<int32_t>(static_cast<int8_t>(value + 128)),
				    static_cast<int64_t>(time.value()), static_cast<int64_t>(b),
				    static_cast<int64_t>(atomic_neuron_on_network.neuron_on_population)});
			}
		}
	}

	// handle
	for (auto const& [descriptor, s] : samples) {
		ret[descriptor] = CADCHandle(
		    std::move(s), static_cast<int>(data.batch_size()),
		    static_cast<int>(std::get<Population>(
		                         network_graph.get_network()
		                             ->execution_instances.at(descriptor.toExecutionInstanceID())
		                             .populations.at(descriptor.toPopulationOnExecutionInstance()))
		                         .neurons.size()));
	}

	return ret;
}

} // namespace hxtorch::spiking
