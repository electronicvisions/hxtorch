#include "hxtorch/snn/run.h"

#include "grenade/vx/network/placed_logical/run.h"
#include "hxtorch/detail/connection.h"

namespace hxtorch::snn {

grenade::vx::signal_flow::IODataMap run(
    lola::vx::v3::Chip const& config,
    grenade::vx::network::placed_logical::NetworkGraph const& network_graph,
    grenade::vx::signal_flow::IODataMap const& inputs,
    grenade::vx::signal_flow::ExecutionInstancePlaybackHooks& playback_hooks)
{
	if (!hxtorch::detail::getExecutor()) {
		throw std::runtime_error("No connection present.");
	}

	return grenade::vx::network::placed_logical::run(
	    *detail::getExecutor(), config, network_graph, inputs, playback_hooks);
}

}
