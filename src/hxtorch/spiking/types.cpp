#include "hxtorch/spiking/types.h"
#include "hxtorch/spiking/detail/to_dense.h"
#include <ATen/Functions.h>
#include <ATen/SparseTensorUtils.h>
#include <log4cxx/logger.h>

namespace hxtorch::spiking {

DataHandle::DataHandle(){};

DataHandle::DataHandle(at::sparse::SparseTensor data, float dt) : m_data(data), m_dt(dt){};


float DataHandle::get_dt()
{
	return m_dt;
};


void DataHandle::set_data(at::sparse::SparseTensor data, float dt)
{
	m_data = data;
	m_dt = dt;
};


at::sparse::SparseTensor DataHandle::get_data()
{
	return m_data;
};


// Specialization for SpikeHandle
torch::Tensor SpikeHandle::to_dense(float dt)
{
	return detail::sparse_spike_to_dense(get_data(), get_dt(), dt);
};


// Specialization for CADCHandle
torch::Tensor CADCHandle::to_dense(float dt, std::string mode)
{
	// TODO: Use this with overloading or enums/switch?
	if (mode == "linear") {
		// Use linear interpolation
		return detail::sparse_cadc_to_dense_linear(get_data(), get_dt(), dt);
	}
	if (mode == "nn") {
		// Use nearest neighbor interpolation
		return detail::sparse_cadc_to_dense_nn(get_data(), get_dt(), dt);
	}

	std::stringstream ss;
	ss << "Mode '" << mode << "' is not supported yet. Supported modes are: 'linear', 'nn'.";
	throw std::runtime_error(ss.str());
};


// Specialization for CADCHandle returning raw data
std::tuple<torch::Tensor, torch::Tensor> CADCHandle::to_raw()
{
	return detail::sparse_cadc_to_dense_raw(get_data());
};


// Dense tensor return if not dt is given
std::tuple<torch::Tensor, float> CADCHandle::to_dense(std::string mode)
{
	auto const& data = get_data();
	auto const& col_data = data.coalesce();
	float dt =
	    static_cast<float>(col_data.indices()[1].max().item<int>()) *
	    (static_cast<float>(data.sizes()[0] * data.sizes()[2]) /
	     static_cast<float>(col_data.values().sizes()[0] - data.sizes()[0] * data.sizes()[2])) *
	    get_dt();

	return std::make_tuple(to_dense(dt, mode), dt);
}


} // namespace hxtorch::spiking
