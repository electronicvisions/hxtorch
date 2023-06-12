#include "hxtorch/core/weight_to_connection.h"
#include "grenade/vx/network/projection.h"
#include "lola/vx/v3/synapse.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace hxtorch::core {

/** Transform weight tensor to grenade connections. Note all entries must be of the same receptor
 * type.
 */
grenade::vx::network::Projection::Connections weight_to_connection(pybind11::array_t<int> weight)
{
	grenade::vx::network::Projection::Connections connections;
	auto weight_data = weight.unchecked<2>();

	for (pybind11::ssize_t row = 0; row < weight.shape(0); ++row) {
		for (pybind11::ssize_t col = 0; col < weight.shape(1); ++col) {
			connections.push_back(
			    {grenade::vx::network::Projection::Connection::Index{
			         static_cast<size_t>(col),
			         halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron()},
			     grenade::vx::network::Projection::Connection::Index{
			         static_cast<size_t>(row),
			         halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron()},
			     grenade::vx::network::Projection::Connection::Weight(
			         std::abs(weight_data(row, col)))});
		}
	}

	return connections;
}


/** Specialization for sparse projections with connection list
 */
grenade::vx::network::Projection::Connections weight_to_connection(
    pybind11::array_t<int> weight, std::vector<std::vector<int>> connections)
{
	if (weight.ndim() != 1) {
		throw std::runtime_error("Only 1D weight tensors are supported.");
	}
	if (weight.shape()[0] != static_cast<int>(connections.at(0).size())) {
		throw std::runtime_error("Number of weights does not match number of connections.");
	}
	if (connections.size() != 2) {
		throw std::runtime_error("Only 2D connection lists are supported.");
	}
	if (connections.at(0).size() != connections.at(1).size()) {
		throw std::runtime_error("Connection lists must have same size along second dimension.");
	}

	grenade::vx::network::Projection::Connections gconnections;
	auto weight_data = weight.unchecked<1>();

	for (int i = 0; i < static_cast<int>(connections.at(0).size()); ++i) {
		auto const& pre = connections.at(1).at(i);
		auto const& post = connections.at(0).at(i);
		gconnections.push_back(
		    {grenade::vx::network::Projection::Connection::Index{
		         static_cast<size_t>(pre), halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron()},
		     grenade::vx::network::Projection::Connection::Index{
		         static_cast<size_t>(post),
		         halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron()},
		     grenade::vx::network::Projection::Connection::Weight(std::abs(weight_data(i)))});
	}

	return gconnections;
}

} // namespace hxtorch::core
