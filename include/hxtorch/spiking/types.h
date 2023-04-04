#pragma once
#include <string>
#include <tuple>
#include <ATen/SparseTensorUtils.h>
#include <torch/torch.h>

namespace hxtorch::spiking {

class DataHandle
{
private:
	/* Data tensor
	 * `at::sparse::SparseTensor` is alias for `torch::Tensor`
	 */
	at::sparse::SparseTensor m_data;

	/* Giving the temporal resolution of the data hold
	 */
	float m_dt;

public:
	/* Default constructor without time resolution
	 */
	DataHandle();

	/* Constructor with time resolution
	 * @param data Sparse data tensor assigned to handle
	 * @param dt Time resolution of data
	 */
	DataHandle(at::sparse::SparseTensor data, float dt);

	/* Getter for time resolution dt
	 */
	float get_dt();

	/* Setter for sparse data tensor
	 * @param data Sparse data tensor assigned to handle
	 * @param dt Time resolution of data
	 */
	void set_data(at::sparse::SparseTensor data, float dt);

	/* Getter for sparse data tensor
	 */
	at::sparse::SparseTensor get_data();
};


/*
 * Specialization for spike data handle since we to_dense behaves differently
 */
class SpikeHandle : public DataHandle
{
public:
	using DataHandle::DataHandle;

	/* Transform the sparse data into a dense tensor
	 * @param dt Desired temporal resoltion of dense tensor
	 */
	torch::Tensor to_dense(float dt);
};


/*
 * Specialization for madc data handle since we to_dense behaves differently
 */
class MADCHandle : public DataHandle
{
public:
	using DataHandle::DataHandle;
};


/*
 * Specialization for cadc data handle since we to_dense behaves differently
 */
class CADCHandle : public DataHandle
{
public:
	using DataHandle::DataHandle;

	/* Transform the sparse data into a dense tensor
	 * @param dt Desired temporal resoltion of dense tensor
	 * @param mode The mode used to transform the sparse CADC tensor to a dense tensor. Currently
	 * supported: 'linear'.
	 */
	torch::Tensor to_dense(float dt, std::string mode = "linear");

	/* Transform the sparse data into a dense tensor representing the hardware data as close as
	 * possible.
	 * @param mode The mode used for interpoaltion.
	 * @returns Returns the dense tensor together with the assumed temporal resolution given by the
	 * average time step.
	 */
	std::tuple<torch::Tensor, float> to_dense(std::string mode = "linear");

	/* Transform the sparse data into a dense tensor holding raw data
	 * @returns Returns a tuple of a tensor holding the CADC data and a tensor holding the
	 * corresponding timestamps.
	 */
	std::tuple<torch::Tensor, torch::Tensor> to_raw();
};


} // namespace::spiking
