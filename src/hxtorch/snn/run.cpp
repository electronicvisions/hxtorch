#include "hxtorch/snn/run.h"
#include "grenade/vx/network/run.h"
#include "hxtorch/detail/connection.h"

namespace hxtorch::snn {

grenade::vx::IODataMap run(
    lola::vx::v3::Chip const& config,
    grenade::vx::network::NetworkGraph const& network_graph,
    grenade::vx::IODataMap const& inputs,
    grenade::vx::ExecutionInstancePlaybackHooks& playback_hooks)
{
	if (!hxtorch::detail::getConnection()) {
		throw std::runtime_error("No connection present.");
	}

	return grenade::vx::network::run(
	    *hxtorch::detail::getConnection(), config, network_graph, inputs, playback_hooks);
}

}
