#include <gtest/gtest.h>

#include "hxtorch/perceptron/detail/iterator.h"

TEST(MultidimIterator, General)
{
	std::vector<int64_t> const range = {2, 3, 4};

	hxtorch::perceptron::detail::MultidimIterator iterator(range);

	std::vector<std::vector<int64_t>> actual;
	for (; iterator != iterator.end(); ++iterator) {
		actual.push_back(*iterator);
	}

	std::vector<std::vector<int64_t>> const expectation = {
	    {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 0, 3}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 1, 3},
	    {0, 2, 0}, {0, 2, 1}, {0, 2, 2}, {0, 2, 3}, {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 0, 3},
	    {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 1, 3}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2}, {1, 2, 3}};

	EXPECT_EQ(actual, expectation);
}
