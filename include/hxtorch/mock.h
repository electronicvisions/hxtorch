#pragma once

#include "hxtorch/constants.h"

namespace hxtorch {

/**
 * Parameter of hardware mock.
 */
struct MockParameter
{
	/* Standard deviation of noise to add to membrane potentials. */
	double noise_std = constants::defaults::noise_std;
	/* Gain to assume for analog multiplication */
	double gain = constants::defaults::gain;

	/** Default constructor. */
	MockParameter() = default;

	/**
	 * Construct with noise standard deviation and gain.
	 *
	 * @param noise_std Noise standard deviation to use
	 * @param gain Gain to use
	 */
	MockParameter(double noise_std, double gain) : noise_std(noise_std), gain(gain) {}
};

/*
 * Returns the current mock parameters.
 */
MockParameter get_mock_parameter();

/*
 * Sets the mock parameters.
 */
void set_mock_parameter(MockParameter const& parameter);

/*
 * Measures the mock parameters, i.e. gain and noise_std, by multiplying a
 * full weight with an artificial test input on the BSS-2 chip.
 * For this purpose a random pattern is used, whose mean value is successively
 * reduced to also work with higher gain factors.
 * The output for the actual calibration is chosen such that it is close to
 * the middle of the available range.
 */
MockParameter measure_mock_parameter();

} // namespace hxtorch
