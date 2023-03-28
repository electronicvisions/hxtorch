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

} // namespace hxtorch::core
