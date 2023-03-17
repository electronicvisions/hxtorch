#pragma once
#include "grenade/vx/network/placed_logical/network_graph.h"
#include "grenade/vx/signal_flow/execution_instance_playback_hooks.h"
#include "grenade/vx/signal_flow/io_data_map.h"
#include "lola/vx/v3/chip.h"


namespace hxtorch::snn {

/**
 * Strips connection from grenade::vx::network::placed_logical::run for python exposure
 */
grenade::vx::signal_flow::IODataMap run(
    lola::vx::v3::Chip const& config,
    grenade::vx::network::placed_logical::NetworkGraph const& network_graph,
    grenade::vx::signal_flow::IODataMap const& inputs,
    grenade::vx::signal_flow::ExecutionInstancePlaybackHooks& playback_hooks);
}
