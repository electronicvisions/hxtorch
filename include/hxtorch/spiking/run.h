#pragma once
#include "grenade/vx/execution/jit_graph_executor.h"
#include "grenade/vx/network/network_graph.h"
#include "grenade/vx/signal_flow/io_data_map.h"
#include "lola/vx/v3/chip.h"


namespace hxtorch::spiking {

/**
 * Strips connection from grenade::vx::network::run for python exposure
 */
grenade::vx::signal_flow::IODataMap run(
    grenade::vx::execution::JITGraphExecutor::ChipConfigs const& config,
    grenade::vx::network::NetworkGraph const& network_graph,
    grenade::vx::signal_flow::IODataMap const& inputs,
    grenade::vx::execution::JITGraphExecutor::PlaybackHooks& playback_hooks);
} // namespace hxtorch::spiking
