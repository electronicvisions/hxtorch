#include "grenade/vx/network/projection.h"
#include <pybind11/numpy.h>

namespace hxtorch::core {

/**
 * Turns a weight matrix given as a rectangular NumPy array of type int into grenade connections.
 * Each entry in the weight matrix is translated to a single connection. The entries are expected to
 * be positive integers.
 *
 * @param weight NumPy tensor holding the weights as positive integers.
 * @return All grenade connections given as a vector of connections.
 */
grenade::vx::network::Projection::Connections weight_to_connection(pybind11::array_t<int> weight)
    SYMBOL_VISIBLE;

} // namespace hxtorch::core
