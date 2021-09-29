#pragma once
#include "grenade/vx/compute/mac.h"
#include "grenade/vx/types.h"
#include "halco/hicann-dls/vx/v2/synapse.h"
#include <limits>

namespace hxtorch::constants {

constexpr static intmax_t synaptic_weight_min = grenade::vx::compute::MAC::Weight::min;
constexpr static intmax_t synaptic_weight_max = grenade::vx::compute::MAC::Weight::max;

constexpr static intmax_t input_activation_min = grenade::vx::UInt5::min;
constexpr static intmax_t input_activation_max = grenade::vx::UInt5::max;

constexpr static intmax_t output_activation_min =
    std::numeric_limits<grenade::vx::Int8::value_type>::min();
constexpr static intmax_t output_activation_max =
    std::numeric_limits<grenade::vx::Int8::value_type>::max();

constexpr static intmax_t hardware_matrix_height =
    halco::hicann_dls::vx::v2::SynapseRowOnSynram::size / 2; // signed weight needs two rows
constexpr static intmax_t hardware_matrix_width =
    halco::hicann_dls::vx::v2::SynapseOnSynapseRow::size;

} // hxtorch::constants
