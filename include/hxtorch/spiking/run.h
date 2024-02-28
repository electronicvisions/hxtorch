#pragma once
#include "grenade/vx/execution/jit_graph_executor.h"
#include "grenade/vx/network/network_graph.h"
#include "grenade/vx/signal_flow/input_data.h"
#include "grenade/vx/signal_flow/output_data.h"
#include "lola/vx/v3/chip.h"


namespace hxtorch::spiking {

/**
 * Strips connection from grenade::vx::network::run for python exposure
 */
grenade::vx::signal_flow::OutputData run(
    grenade::vx::execution::JITGraphExecutor::ChipConfigs const& config,
    grenade::vx::network::NetworkGraph const& network_graph,
    grenade::vx::signal_flow::InputData const& inputs,
    grenade::vx::execution::JITGraphExecutor::Hooks& hooks);
} // namespace hxtorch::spiking
