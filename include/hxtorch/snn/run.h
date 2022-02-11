#pragma once
#include "grenade/vx/config.h"
#include "grenade/vx/execution_instance_playback_hooks.h"
#include "grenade/vx/io_data_map.h"
#include "grenade/vx/network/network_graph.h"


namespace hxtorch::snn {

/**
 * Strips connection from grenade::vx::network::run for python exposure
 */
grenade::vx::IODataMap run(
    grenade::vx::ChipConfig const& config,
    grenade::vx::network::NetworkGraph const& network_graph,
    grenade::vx::IODataMap const& inputs,
    grenade::vx::ExecutionInstancePlaybackHooks& playback_hooks);

}
