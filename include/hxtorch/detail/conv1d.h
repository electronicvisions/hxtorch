#pragma once

#include <torch/torch.h>

namespace hxtorch::detail {

size_t conv1d_num_outputs(size_t x_size, size_t weights_size, size_t stride);

} // namespace hxtorch::detail
