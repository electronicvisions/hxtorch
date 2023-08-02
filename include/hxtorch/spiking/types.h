#pragma once
#include <string>
#include <tuple>
#include <torch/torch.h>

namespace hxtorch::spiking {

template <typename T>
class DataHandle
{
private:
	T m_data;
	int m_batch_size;
	int m_population_size;

public:
	DataHandle() : m_batch_size(0), m_population_size(0){};
	/* Constructor with time resolution
	 * @param data Sparse data tensor assigned to handle
	 */
	template <typename U>
	DataHandle(U&& data, int batch_size, int population_size) :
	    m_data(std::forward<U>(data)), m_batch_size(batch_size), m_population_size(population_size)
	{}

	/* Setter for sparse data tensor
	 * @param data Sparse data tensor assigned to handle
	 * @param dt Time resolution of data
	 */
	template <typename U>
	void set_data(U&& data, int batch_size, int population_size)
	{
		m_data = std::forward<U>(data);
		m_batch_size = batch_size;
		m_population_size = population_size;
	}
	/* Getter for sparse data tensor
	 */
	T& get_data()
	{
		return m_data;
	};

	int batch_size()
	{
		return m_batch_size;
	};
	int population_size()
	{
		return m_population_size;
	};
};


/*
 * Specialization for spike data handle since we to_dense behaves differently
 */
class SpikeHandle : public DataHandle<std::vector<std::tuple<int64_t, int64_t, int64_t>>>
{
public:
	using DataHandle::DataHandle;

	/* Transform the sparse data into a dense tensor
	 * @param dt Desired temporal resoltion of dense tensor
	 */
	torch::Tensor to_dense(float runtime, float dt);
};


/*
 * Specialization for madc data handle since we to_dense behaves differently
 */
class MADCHandle : public DataHandle<std::vector<std::tuple<int16_t, int64_t, int64_t, int64_t>>>
{
public:
	using DataHandle::DataHandle;
};


/*
 * Specialization for cadc data handle since we to_dense behaves differently
 */
class CADCHandle : public DataHandle<std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>>>
{
public:
	using DataHandle::DataHandle;

	/* Transform the sparse data into a dense tensor
	 * @param dt Desired temporal resoltion of dense tensor
	 * @param mode The mode used to transform the sparse CADC tensor to a dense tensor. Currently
	 * supported: 'linear'.
	 */
	torch::Tensor to_dense(float runtime, float dt, std::string mode = "linear");

	/* Transform the sparse data into a dense tensor representing the hardware data as close as
	 * possible.
	 * @param mode The mode used for interpoaltion.
	 * @returns Returns the dense tensor together with the assumed temporal resolution given by the
	 * average time step.
	 */
	std::tuple<torch::Tensor, float> to_dense(float runtime, std::string mode = "linear");

	/* Transform the sparse data into a dense tensor holding raw data
	 * @returns Returns a tuple of a tensor holding the CADC data and a tensor holding the
	 * corresponding timestamps.
	 */
	std::tuple<torch::Tensor, torch::Tensor> to_raw();
};


} // namespace::spiking
