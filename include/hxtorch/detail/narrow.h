#pragma once
#include <array>

#include <torch/torch.h>

namespace hxtorch::detail {

template <typename T>
auto multi_narrow(
    T& t, std::vector<int64_t> dim, std::vector<int64_t> start, std::vector<int64_t> length)
{
	assert(dim.size() == start.size() && dim.size() == length.size());
	if (dim.size() == 0) {
		return t;
	} else {
		auto const d = dim.back();
		auto const s = start.back();
		auto const l = length.back();
		dim.pop_back();
		start.pop_back();
		length.pop_back();
		return multi_narrow(t, dim, start, length).narrow(d, s, l);
	}
}

} // namespace hxtorch::detail
