#include "hxtorch/snn/run.h"

#include "grenade/vx/network/placed_atomic/run.h"
#include "hxtorch/detail/connection.h"

namespace hxtorch::snn {

grenade::vx::signal_flow::IODataMap run(
    lola::vx::v3::Chip const& config,
    grenade::vx::network::placed_atomic::NetworkGraph const& network_graph,
    grenade::vx::signal_flow::IODataMap const& inputs,
    grenade::vx::signal_flow::ExecutionInstancePlaybackHooks& playback_hooks)
{
	if (!hxtorch::detail::getConnection()) {
		throw std::runtime_error("No connection present.");
	}

	return grenade::vx::network::placed_atomic::run(
	    *detail::getConnection(), config, network_graph, inputs, playback_hooks);
}

}
