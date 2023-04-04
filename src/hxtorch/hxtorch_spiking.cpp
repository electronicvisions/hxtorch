#include <torch/torch.h>

#include <torch/custom_class.h>
#include <torch/script.h>
// for proper handling of torch's python tensor to C++ torch::Tensor conversion
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/extension.h>

#include "hxtorch/spiking/docstrings.h"
#include "hxtorch/spiking/extract_tensors.h"
#include "hxtorch/spiking/run.h"
#include "hxtorch/spiking/tensor_to_spike_times.h"
#include "hxtorch/spiking/weight_to_connection.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.import("pygrenade_vx");
	pybind11::class_<hxtorch::spiking::DataHandle>(m, "DataHandle")
	    .def(pybind11::init<torch::Tensor, float>(), pybind11::arg("data"), pybind11::arg("dt"))
	    .def_property(
	        "data", &hxtorch::spiking::DataHandle::get_data,
	        &hxtorch::spiking::DataHandle::set_data);
	pybind11::class_<hxtorch::spiking::SpikeHandle>(m, "SpikeHandle")
	    .def(pybind11::init<torch::Tensor, float>(), pybind11::arg("data"), pybind11::arg("dt"))
	    .def("get_data", &hxtorch::spiking::SpikeHandle::get_data)
	    .def(
	        "set_data", &hxtorch::spiking::DataHandle::set_data, pybind11::arg("data"),
	        pybind11::arg("dt"))
	    .def(
	        "to_dense", static_cast<torch::Tensor (hxtorch::spiking::SpikeHandle::*)(float)>(
	                        &hxtorch::spiking::SpikeHandle::to_dense));
	pybind11::class_<hxtorch::spiking::CADCHandle>(m, "CADCHandle")
	    .def(pybind11::init<torch::Tensor, float>(), pybind11::arg("data"), pybind11::arg("dt"))
	    .def("get_data", &hxtorch::spiking::CADCHandle::get_data)
	    .def(
	        "set_data", &hxtorch::spiking::CADCHandle::set_data, pybind11::arg("data"),
	        pybind11::arg("dt"))
	    .def(
	        "to_dense",
	        static_cast<torch::Tensor (hxtorch::spiking::CADCHandle::*)(float, std::string)>(
	            &hxtorch::spiking::CADCHandle::to_dense),
	        pybind11::arg("dt"), pybind11::arg("mode") = "linear")
	    .def(
	        "to_dense",
	        static_cast<std::tuple<torch::Tensor, float> (hxtorch::spiking::CADCHandle::*)(
	            std::string)>(&hxtorch::spiking::CADCHandle::to_dense),
	        pybind11::arg("mode") = "linear")
	    .def("to_raw", &hxtorch::spiking::CADCHandle::to_raw);
	pybind11::class_<hxtorch::spiking::MADCHandle>(m, "MADCHandle")
	    .def(pybind11::init<torch::Tensor, float>(), pybind11::arg("data"), pybind11::arg("dt"))
	    .def_property(
	        "data", &hxtorch::spiking::MADCHandle::get_data,
	        &hxtorch::spiking::MADCHandle::set_data);
	m.def(
	    "run", &hxtorch::spiking::run, pybind11::arg("config"), pybind11::arg("network_graph"),
	    pybind11::arg("inputs"), pybind11::arg("playback_hooks"));
	m.def("weight_to_connection", &hxtorch::spiking::weight_to_connection, pybind11::arg("weight"));

	m.def(
	    "tensor_to_spike_times", &hxtorch::spiking::tensor_to_spike_times, pybind11::arg("times"),
	    pybind11::arg("dt"));
	m.def(
	    "extract_spikes", &hxtorch::spiking::extract_spikes, pybind11::arg("data"),
	    pybind11::arg("network_graph"), pybind11::arg("runtime"));
	m.def(
	    "extract_cadc", &hxtorch::spiking::extract_cadc, pybind11::arg("data"),
	    pybind11::arg("network_graph"), pybind11::arg("runtime"));
	m.def(
	    "extract_madc", &hxtorch::spiking::extract_madc, pybind11::arg("data"),
	    pybind11::arg("network_graph"), pybind11::arg("runtime"));
}
