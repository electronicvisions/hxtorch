#include "hxtorch/detail/conv1d.h"

namespace hxtorch::detail {

size_t conv1d_num_outputs(size_t x_size, size_t weights_size, size_t stride)
{
	return ((x_size - (weights_size - 1) - 1) / stride) + 1;
}

} // namespace hxtorch::detail
