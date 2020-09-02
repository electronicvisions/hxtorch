#pragma once

namespace hxtorch {

/**
 * Parameter of hardware mock.
 */
struct MockParameter
{
	/* Standard deviation of noise to add to membrane potentials. */
	float noise_std = 2.;
	/* Gain to assume for analog multiplication */
	float gain = 0.0012;

	/** Default constructor. */
	MockParameter() = default;

	/**
	 * Construct with noise standard deviation and gain.
	 * @param noise_std Noise standard deviation to use
	 * @param gain Gain to use
	 */
	MockParameter(float noise_std, float gain) : noise_std(noise_std), gain(gain) {}
};

} // namespace hxtorch
