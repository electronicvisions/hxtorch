#include "grenade/vx/network/projection.h"
#include <torch/torch.h>

namespace hxtorch::spiking {

/**
 * Turns a weight matrix given as a rectangular torch tensor into grenade connections. Each entry in
 * the weight matrix is translated to a single connection.
 *
 * @param weight Torch tensor holding the weights.
 * @return All grenade connections given as a vector of connections.
 */
grenade::vx::network::Projection::Connections weight_to_connection(torch::Tensor weight);

} // namespace hxtorch::spiking
