#include "hxtorch/snn/run.h"

#include "grenade/vx/jit_graph_executor.h"
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

	auto& executor = *hxtorch::detail::getConnection();
	auto connection = executor.release_connection(halco::hicann_dls::vx::DLSGlobal());

	auto ret = grenade::vx::network::run(connection, config, network_graph, inputs, playback_hooks);

	executor.acquire_connection(halco::hicann_dls::vx::DLSGlobal(), std::move(connection));

	return ret;
}

}
