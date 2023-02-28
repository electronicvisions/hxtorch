#pragma once
#include "grenade/vx/compute/mac.h"
#include "grenade/vx/signal_flow/types.h"
#include "halco/hicann-dls/vx/v3/synapse.h"
#include <limits>

namespace hxtorch::constants {

constexpr static intmax_t synaptic_weight_min = grenade::vx::compute::MAC::Weight::min;
constexpr static intmax_t synaptic_weight_max = grenade::vx::compute::MAC::Weight::max;

constexpr static intmax_t input_activation_min = grenade::vx::signal_flow::UInt5::min;
constexpr static intmax_t input_activation_max = grenade::vx::signal_flow::UInt5::max;

constexpr static intmax_t output_activation_min =
    std::numeric_limits<grenade::vx::signal_flow::Int8::value_type>::min();
constexpr static intmax_t output_activation_max =
    std::numeric_limits<grenade::vx::signal_flow::Int8::value_type>::max();

constexpr static intmax_t hardware_matrix_height =
    halco::hicann_dls::vx::v3::SynapseRowOnSynram::size / 2; // signed weight needs two rows
constexpr static intmax_t hardware_matrix_width =
    halco::hicann_dls::vx::v3::SynapseOnSynapseRow::size;

namespace defaults {

constexpr static intmax_t wait_between_events = 5;
constexpr static double gain = 0.002;
constexpr static double noise_std = 2.;

} // defaults

} // hxtorch::constants
