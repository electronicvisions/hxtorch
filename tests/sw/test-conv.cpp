#include <gtest/gtest.h>

#include "hxtorch/detail/conv.h"

TEST(conv_fold_input, 1D)
{
	auto const input_vector = torch::arange(20).reshape({1, 1, 20});
	auto const actual = hxtorch::detail::conv_fold_input(input_vector, {5}, {3});
	std::vector<int64_t> const expectation_sizes = {1, 6, 5};
	EXPECT_EQ(actual.sizes().vec(), expectation_sizes);
	auto const expectation = torch::tensor(
	    {{{0, 1, 2, 3, 4},
	      {3, 4, 5, 6, 7},
	      {6, 7, 8, 9, 10},
	      {9, 10, 11, 12, 13},
	      {12, 13, 14, 15, 16},
	      {15, 16, 17, 18, 19}}},
	    input_vector.dtype());
	EXPECT_TRUE((torch::equal(actual, expectation)));
}

TEST(conv_fold_input, Multichannel1D)
{
	auto const input_vector = torch::arange(40).reshape({1, 2, 20});
	auto const actual = hxtorch::detail::conv_fold_input(input_vector, {5}, {3});
	std::vector<int64_t> const expectation_sizes = {1, 6, 10};
	EXPECT_EQ(actual.sizes().vec(), expectation_sizes);
	auto const expectation = torch::tensor(
	    {{{0, 1, 2, 3, 4, 20, 21, 22, 23, 24},
	      {3, 4, 5, 6, 7, 23, 24, 25, 26, 27},
	      {6, 7, 8, 9, 10, 26, 27, 28, 29, 30},
	      {9, 10, 11, 12, 13, 29, 30, 31, 32, 33},
	      {12, 13, 14, 15, 16, 32, 33, 34, 35, 36},
	      {15, 16, 17, 18, 19, 35, 36, 37, 38, 39}}},
	    input_vector.dtype());
	EXPECT_TRUE((torch::equal(actual, expectation)));
}

TEST(conv_fold_input, 2D)
{
	auto const input_vector = torch::arange(200).reshape({1, 1, 10, 20});
	auto const actual = hxtorch::detail::conv_fold_input(input_vector, {5, 2}, {3, 4});
	std::vector<int64_t> const expectation_sizes = {1, 10, 10};
	EXPECT_EQ(actual.sizes().vec(), expectation_sizes);
	auto const expectation = torch::tensor(
	    {{{0, 1, 20, 21, 40, 41, 60, 61, 80, 81},
	      {4, 5, 24, 25, 44, 45, 64, 65, 84, 85},
	      {8, 9, 28, 29, 48, 49, 68, 69, 88, 89},
	      {12, 13, 32, 33, 52, 53, 72, 73, 92, 93},
	      {16, 17, 36, 37, 56, 57, 76, 77, 96, 97},
	      {60, 61, 80, 81, 100, 101, 120, 121, 140, 141},
	      {64, 65, 84, 85, 104, 105, 124, 125, 144, 145},
	      {68, 69, 88, 89, 108, 109, 128, 129, 148, 149},
	      {72, 73, 92, 93, 112, 113, 132, 133, 152, 153},
	      {76, 77, 96, 97, 116, 117, 136, 137, 156, 157}}},
	    input_vector.dtype());
	EXPECT_TRUE((torch::equal(actual, expectation)));
}

TEST(conv_fold_input, Multichannel2D)
{
	auto const input_vector = torch::arange(400).reshape({1, 2, 10, 20});
	auto const actual = hxtorch::detail::conv_fold_input(input_vector, {5, 2}, {3, 4});
	std::vector<int64_t> const expectation_sizes = {1, 10, 20};
	EXPECT_EQ(actual.sizes().vec(), expectation_sizes);
	auto const expectation = torch::tensor(
	    {{{0, 1, 20, 21, 40, 41, 60, 61, 80, 81, 200, 201, 220, 221, 240, 241, 260, 261, 280, 281},
	      {4, 5, 24, 25, 44, 45, 64, 65, 84, 85, 204, 205, 224, 225, 244, 245, 264, 265, 284, 285},
	      {8, 9, 28, 29, 48, 49, 68, 69, 88, 89, 208, 209, 228, 229, 248, 249, 268, 269, 288, 289},
	      {12,  13,  32,  33,  52,  53,  72,  73,  92,  93,
	       212, 213, 232, 233, 252, 253, 272, 273, 292, 293},
	      {16,  17,  36,  37,  56,  57,  76,  77,  96,  97,
	       216, 217, 236, 237, 256, 257, 276, 277, 296, 297},
	      {60,  61,  80,  81,  100, 101, 120, 121, 140, 141,
	       260, 261, 280, 281, 300, 301, 320, 321, 340, 341},
	      {64,  65,  84,  85,  104, 105, 124, 125, 144, 145,
	       264, 265, 284, 285, 304, 305, 324, 325, 344, 345},
	      {68,  69,  88,  89,  108, 109, 128, 129, 148, 149,
	       268, 269, 288, 289, 308, 309, 328, 329, 348, 349},
	      {72,  73,  92,  93,  112, 113, 132, 133, 152, 153,
	       272, 273, 292, 293, 312, 313, 332, 333, 352, 353},
	      {76,  77,  96,  97,  116, 117, 136, 137, 156, 157,
	       276, 277, 296, 297, 316, 317, 336, 337, 356, 357}}},
	    input_vector.dtype());
	EXPECT_TRUE((torch::equal(actual, expectation)));
}

TEST(conv_unfold_input, 1D)
{
	auto const input_vector = torch::rand({1, 1, 21}, torch::kFloat32);
	auto const folded = hxtorch::detail::conv_fold_input(input_vector, {5}, {3});
	auto const actual = hxtorch::detail::conv_unfold_input(folded, 1, {5}, {6}, {3});
	std::vector<int64_t> const expectation_sizes = {1, 1, 20};
	EXPECT_EQ(actual.sizes().vec(), expectation_sizes);
	EXPECT_TRUE((torch::equal(actual, input_vector.narrow(2, 0, 20))));
}

TEST(conv_unfold_input, Multichannel1D)
{
	auto const input_vector = torch::rand({1, 2, 20}, torch::kFloat32);
	auto const folded = hxtorch::detail::conv_fold_input(input_vector, {5}, {3});
	auto const actual = hxtorch::detail::conv_unfold_input(folded, 2, {5}, {6}, {3});
	std::vector<int64_t> const expectation_sizes = {1, 2, 20};
	EXPECT_EQ(actual.sizes().vec(), expectation_sizes);
	EXPECT_TRUE((torch::equal(actual, input_vector.narrow(2, 0, 20))));
}

TEST(conv_unfold_input, 2D)
{
	auto const input_vector = torch::rand({1, 1, 10, 20}, torch::kFloat32);
	auto const folded = hxtorch::detail::conv_fold_input(input_vector, {5, 2}, {3, 2});
	auto const actual = hxtorch::detail::conv_unfold_input(folded, 1, {5, 2}, {2, 10}, {3, 2});
	std::vector<int64_t> const expectation_sizes = {1, 1, 8, 20};
	EXPECT_EQ(actual.sizes().vec(), expectation_sizes);
	EXPECT_TRUE((torch::equal(actual, input_vector.narrow(2, 0, 8).narrow(3, 0, 20))));
}

TEST(conv_unfold_input, Multichannel2D)
{
	auto const input_vector = torch::rand({1, 2, 10, 20}, torch::kFloat32);
	auto const folded = hxtorch::detail::conv_fold_input(input_vector, {5, 2}, {3, 2});
	auto const actual = hxtorch::detail::conv_unfold_input(folded, 2, {5, 2}, {2, 10}, {3, 2});
	std::vector<int64_t> const expectation_sizes = {1, 2, 8, 20};
	EXPECT_EQ(actual.sizes().vec(), expectation_sizes);
	EXPECT_TRUE((torch::equal(actual, input_vector.narrow(2, 0, 8).narrow(3, 0, 20))));
}

TEST(conv_fold_output, General)
{
	auto const input = torch::arange(24).reshape({2, 1, 3, 4});
	auto const actual = hxtorch::detail::conv_fold_output(input);
	std::vector<int64_t> const expectation_sizes = {2, 1, 12};
	EXPECT_EQ(actual.sizes().vec(), expectation_sizes);
	auto const expectation = torch::tensor(
	    {{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}},
	     {{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}}},
	    input.dtype());
	EXPECT_TRUE((torch::equal(actual, expectation)));
	EXPECT_TRUE((torch::equal(actual, torch::arange(24).reshape({2, 1, 12}))));
}

TEST(conv_unfold_output, General)
{
	auto const input = torch::arange(24).reshape({2, 1, 3, 4});
	auto const folded = hxtorch::detail::conv_fold_output(input);
	auto const actual = hxtorch::detail::conv_unfold_output(folded, {3, 4});
	std::vector<int64_t> const expectation_sizes = {2, 1, 12};
	EXPECT_EQ(actual.sizes().vec(), input.sizes().vec());
	EXPECT_TRUE((torch::equal(actual, input)));
}

TEST(conv_fold_kernel, General)
{
	auto const input = torch::arange(24).reshape({2, 3, 4});
	auto const actual = hxtorch::detail::conv_fold_kernel(input);
	std::vector<int64_t> const expectation_sizes = {12, 2};
	EXPECT_EQ(actual.sizes().vec(), expectation_sizes);
	auto const expectation = torch::tensor(
	                             {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
	                              {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}},
	                             input.dtype())
	                             .t();
	EXPECT_TRUE((torch::equal(actual, expectation)));
	EXPECT_TRUE((torch::equal(actual, torch::arange(24).reshape({2, 12}).t())));
}

TEST(conv_unfold_kernel, General)
{
	auto const input = torch::arange(120).reshape({2, 3, 4, 5});
	auto const folded = hxtorch::detail::conv_fold_kernel(input);
	auto const actual = hxtorch::detail::conv_unfold_kernel(folded, {4, 5}, 3);
	EXPECT_EQ(actual.sizes().vec(), input.sizes().vec());
	EXPECT_TRUE((torch::equal(actual, input)));
}

TEST(conv_permute_output, General)
{
	auto const input = torch::rand({1, 2, 3, 4});
	auto const actual = hxtorch::detail::conv_permute_output(input);
	std::vector<int64_t> const expectation_sizes = {1, 4, 2, 3};
	EXPECT_EQ(actual.sizes().vec(), expectation_sizes);
}

TEST(conv_unpermute_output, General)
{
	auto const input = torch::rand({1, 2, 3, 4});
	auto const actual = hxtorch::detail::conv_unpermute_output(input);
	std::vector<int64_t> const expectation_sizes = {1, 3, 4, 2};
	EXPECT_EQ(actual.sizes().vec(), expectation_sizes);
}
