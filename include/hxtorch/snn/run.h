#pragma once
#include "grenade/vx/execution_instance_playback_hooks.h"
#include "grenade/vx/io_data_map.h"
#include "grenade/vx/network/network_graph.h"
#include "lola/vx/v2/chip.h"


namespace hxtorch::snn {

/**
 * Strips connection from grenade::vx::network::run for python exposure
 */
grenade::vx::IODataMap run(
    lola::vx::v2::Chip const& config,
    grenade::vx::network::NetworkGraph const& network_graph,
    grenade::vx::IODataMap const& inputs,
    grenade::vx::ExecutionInstancePlaybackHooks& playback_hooks);
}
