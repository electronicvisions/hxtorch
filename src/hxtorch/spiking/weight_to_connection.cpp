#include "hxtorch/spiking/weight_to_connection.h"
#include "grenade/vx/network/placed_logical/projection.h"
#include "lola/vx/v3/synapse.h"

#include <ATen/ATen.h>
#include <torch/torch.h>


namespace hxtorch::spiking {

/** Transform weight tensor to grenade connections. Note all entries must be of the same receptor
 * type.
 */
grenade::vx::network::placed_logical::Projection::Connections weight_to_connection(
    torch::Tensor weight)
{
	if (weight.dim() != 2) {
		throw std::runtime_error("Only 2D weight tensors are supported.");
	}

	// entries must have same signs or must be zero
	auto const signs = torch::sign(weight);
	// TODO: Maybe support CUDA
	auto const [unique_signs, _] = at::_unique(signs, false, false);
	if ((unique_signs.sizes()[0] == 3) ||
	    ((unique_signs.sizes()[0] == 2) && (unique_signs[0].item().toInt() != 0) &&
	     (unique_signs[-1].item().toInt() != 0))) {
		throw std::runtime_error(
		    "Encountered different signs in the weights. Only one sign (or zero) is supported.");
	}

	grenade::vx::network::placed_logical::Projection::Connections connections;
	for (int row = 0; row < weight.sizes()[0]; ++row) {
		for (int col = 0; col < weight.sizes()[1]; ++col) {
			auto w = weight.index({row, col}).item().toInt();
			connections.push_back(
			    {grenade::vx::network::placed_logical::Projection::Connection::Index{
			         static_cast<size_t>(col),
			         halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron()},
			     grenade::vx::network::placed_logical::Projection::Connection::Index{
			         static_cast<size_t>(row),
			         halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron()},
			     grenade::vx::network::placed_logical::Projection::Connection::Weight(
			         std::abs(w))});
		}
	}

	return connections;
}
}
