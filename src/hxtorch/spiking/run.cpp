#include "hxtorch/spiking/run.h"

#include "grenade/vx/network/run.h"
#include "hxtorch/core/detail/connection.h"

namespace hxtorch::spiking {

grenade::vx::signal_flow::OutputData run(
    grenade::vx::execution::JITGraphExecutor::ChipConfigs const& config,
    grenade::vx::network::NetworkGraph const& network_graph,
    grenade::vx::signal_flow::InputData const& inputs,
    grenade::vx::execution::JITGraphExecutor::PlaybackHooks& playback_hooks)
{
	if (!hxtorch::core::detail::getExecutor()) {
		throw std::runtime_error("No connection present.");
	}

	return grenade::vx::network::run(
	    *hxtorch::core::detail::getExecutor(), network_graph, config, inputs, std::move(playback_hooks));
}

}
