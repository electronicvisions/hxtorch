#pragma once

namespace hxtorch {

/**
 * Parameter of hardware mock.
 */
struct MockParameter
{
	/* Standard deviation of noise to add to membrane potentials. */
	double noise_std = 2.;
	/* Gain to assume for analog multiplication */
	double gain = 0.002;

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

} // namespace hxtorch
