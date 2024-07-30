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
#include "hxtorch/spiking/types.h"

#include "grenade/vx/signal_flow/output_data.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	typedef std::vector<std::tuple<int64_t, int64_t, int64_t>> spike_type;
	typedef std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>> cadc_type;
	typedef std::vector<std::tuple<int16_t, int64_t, int64_t, int64_t>> madc_type;
	m.import("pygrenade_vx");
	pybind11::class_<hxtorch::spiking::SpikeHandle>(m, "SpikeHandle")
	    .def(
	        pybind11::init<spike_type, int, int>(), pybind11::arg("data"),
	        pybind11::arg("batch_size"), pybind11::arg("population_size"))
	    .def_property_readonly("batch_size", &hxtorch::spiking::SpikeHandle::batch_size)
	    .def_property_readonly("population_size", &hxtorch::spiking::SpikeHandle::population_size)
	    .def("get_data", &hxtorch::spiking::SpikeHandle::get_data)
	    .def(
	        "set_data",
	        [](hxtorch::spiking::SpikeHandle& handle, spike_type const& data, int batch_size,
	           int population_size) { handle.set_data(data, batch_size, population_size); },
	        pybind11::arg("data"), pybind11::arg("batch_size"), pybind11::arg("population_size"))
	    .def(
	        "to_dense", static_cast<torch::Tensor (hxtorch::spiking::SpikeHandle::*)(float, float)>(
	                        &hxtorch::spiking::SpikeHandle::to_dense));
	pybind11::class_<hxtorch::spiking::CADCHandle>(m, "CADCHandle")
	    .def(
	        pybind11::init<cadc_type, int, int>(), pybind11::arg("data"),
	        pybind11::arg("batch_size"), pybind11::arg("population_size"))
	    .def_property_readonly("batch_size", &hxtorch::spiking::CADCHandle::batch_size)
	    .def_property_readonly("population_size", &hxtorch::spiking::CADCHandle::population_size)
	    .def("get_data", &hxtorch::spiking::CADCHandle::get_data)
	    .def(
	        "set_data",
	        [](hxtorch::spiking::CADCHandle& handle, cadc_type const& data, int batch_size,
	           int population_size) { handle.set_data(data, batch_size, population_size); },
	        pybind11::arg("data"), pybind11::arg("batch_size"), pybind11::arg("population_size"))
	    .def(
	        "to_dense",
	        static_cast<torch::Tensor (hxtorch::spiking::CADCHandle::*)(float, float, std::string)>(
	            &hxtorch::spiking::CADCHandle::to_dense),
	        pybind11::arg("runtime"), pybind11::arg("dt"), pybind11::arg("mode") = "linear")
	    .def(
	        "to_dense",
	        static_cast<std::tuple<torch::Tensor, float> (hxtorch::spiking::CADCHandle::*)(
	            float, std::string)>(&hxtorch::spiking::CADCHandle::to_dense),
	        pybind11::arg("runtime"), pybind11::arg("mode") = "linear")
	    .def("to_raw", &hxtorch::spiking::CADCHandle::to_raw);
	pybind11::class_<hxtorch::spiking::MADCHandle>(m, "MADCHandle")
	    .def(
	        pybind11::init<madc_type, int, int>(), pybind11::arg("data"),
	        pybind11::arg("batch_size"), pybind11::arg("population_size"))
	    .def_property_readonly("batch_size", &hxtorch::spiking::MADCHandle::batch_size)
	    .def_property_readonly("population_size", &hxtorch::spiking::MADCHandle::population_size)
	    .def("get_data", &hxtorch::spiking::MADCHandle::get_data)
	    .def(
	        "set_data",
	        [](hxtorch::spiking::MADCHandle& handle, madc_type const& data, int batch_size,
	           int population_size) { handle.set_data(data, batch_size, population_size); },
	        pybind11::arg("data"), pybind11::arg("batch_size"), pybind11::arg("population_size"));
	m.def(
	    "run", &hxtorch::spiking::run, pybind11::arg("config"), pybind11::arg("network_graph"),
	    pybind11::arg("inputs"), pybind11::arg("hooks"));
	m.def(
	    "tensor_to_spike_times", &hxtorch::spiking::tensor_to_spike_times, pybind11::arg("times"),
	    pybind11::arg("dt"));
	m.def(
	    "extract_spikes", &hxtorch::spiking::extract_spikes, pybind11::arg("data"),
	    pybind11::arg("network_graph"));
	m.def(
	    "extract_cadc", &hxtorch::spiking::extract_cadc, pybind11::arg("data"),
	    pybind11::arg("network_graph"));
	m.def(
	    "extract_madc", &hxtorch::spiking::extract_madc, pybind11::arg("data"),
	    pybind11::arg("network_graph"));
}
