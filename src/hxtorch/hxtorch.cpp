#include <torch/torch.h>

#include <torch/custom_class.h>
#include <torch/script.h>
// for proper handling of torch's python tensor to C++ torch::Tensor conversion
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/extension.h>

#include "hxtorch/add.h"
#include "hxtorch/argmax.h"
#include "hxtorch/connection.h"
#include "hxtorch/constants.h"
#include "hxtorch/conv.h"
#include "hxtorch/detail/conv.h"
#include "hxtorch/detail/mock.h"
#include "hxtorch/docstrings.h"
#include "hxtorch/inference_tracer.h"
#include "hxtorch/mac.h"
#include "hxtorch/matmul.h"
#include "hxtorch/mock.h"
#include "hxtorch/relu.h"

#include "hxtorch/snn/extract_tensors.h"
#include "hxtorch/snn/run.h"
#include "hxtorch/snn/tensor_to_spike_times.h"
#include "hxtorch/snn/weight_to_connection.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.import("pygrenade_vx");
	m.def(
	    "init_hardware",
	    (void (*)(std::optional<hxtorch::HWDBPath> const&, bool)) & hxtorch::init_hardware,
	    __doc_hxtorch_init_hardware, pybind11::arg("hwdb_path") = std::nullopt,
	    pybind11::arg("spiking") = false);
	m.def(
	    "init_hardware", (void (*)(hxtorch::CalibrationPath const&)) & hxtorch::init_hardware,
	    __doc_hxtorch_init_hardware_2, pybind11::arg("calibration_path"));
	m.def(
	    "init_hardware_minimal", &hxtorch::init_hardware_minimal,
	    __doc_hxtorch_init_hardware_minimal);
	m.def("get_chip", &hxtorch::get_chip, __doc_hxtorch_get_chip);
	m.def("release_hardware", &hxtorch::release_hardware, __doc_hxtorch_release_hardware);
	m.def("get_mock_parameter", &hxtorch::get_mock_parameter, __doc_hxtorch_get_mock_parameter);
	m.def(
	    "set_mock_parameter", &hxtorch::set_mock_parameter, __doc_hxtorch_set_mock_parameter,
	    pybind11::arg("parameter"));
	m.def(
	    "mac", &hxtorch::mac, __doc_hxtorch_mac, pybind11::arg("x"), pybind11::arg("weights"),
	    pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = hxtorch::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	m.def(
	    "measure_mock_parameter", &hxtorch::measure_mock_parameter,
	    __doc_hxtorch_measure_mock_parameter);
	m.def(
	    "relu", &hxtorch::relu, __doc_hxtorch_relu, pybind11::arg("input"),
	    pybind11::arg("mock") = false);
	m.def(
	    "converting_relu", &hxtorch::converting_relu, __doc_hxtorch_converting_relu,
	    pybind11::arg("input"), pybind11::arg("shift") = 2, pybind11::arg("mock") = false);
	m.def(
	    "inference_trace", &hxtorch::inference_trace, __doc_hxtorch_inference_trace,
	    pybind11::arg("input"), pybind11::arg("filename"));
	m.def(
	    "argmax", &hxtorch::argmax, __doc_hxtorch_argmax, pybind11::arg("input"),
	    pybind11::arg("dim") = c10::optional<int64_t>(), pybind11::arg("keepdim") = false,
	    pybind11::arg("mock") = false);
	m.def(
	    "add", &hxtorch::add, __doc_hxtorch_add, pybind11::arg("input"), pybind11::arg("other"),
	    pybind11::arg("alpha") = 1, pybind11::arg("mock") = false);
	m.def(
	    "matmul", &hxtorch::matmul, __doc_hxtorch_matmul, pybind11::arg("input"),
	    pybind11::arg("other"), pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = hxtorch::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);

	typedef torch::Tensor (*single_stride_conv_type)(
	    torch::Tensor const&, torch::Tensor const&, c10::optional<torch::Tensor> const&, int64_t,
	    int64_t, int64_t, bool);
	typedef torch::Tensor (*conv1d_type)(
	    torch::Tensor const&, torch::Tensor const&, c10::optional<torch::Tensor> const&,
	    std::array<int64_t, 1>, int64_t, int64_t, bool);
	typedef torch::Tensor (*conv2d_type)(
	    torch::Tensor const&, torch::Tensor const&, c10::optional<torch::Tensor> const&,
	    std::array<int64_t, 2>, int64_t, int64_t, bool);
	typedef torch::Tensor (*single_stride_expanded_conv1d_type)(
	    torch::Tensor const&, torch::Tensor const&, c10::optional<torch::Tensor> const&, int64_t,
	    int64_t, int64_t, int64_t, bool);
	typedef torch::Tensor (*expanded_conv1d_type)(
	    torch::Tensor const&, torch::Tensor const&, c10::optional<torch::Tensor> const&,
	    std::array<int64_t, 1>, int64_t, int64_t, int64_t, bool);

	m.def(
	    "conv1d", (single_stride_conv_type) &hxtorch::conv1d, __doc_hxtorch_conv1d,
	    pybind11::arg("input"), pybind11::arg("weight"),
	    pybind11::arg("bias") = c10::optional<torch::Tensor>(), pybind11::arg("stride") = 1,
	    pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = hxtorch::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	m.def(
	    "conv1d", (conv1d_type) &hxtorch::conv1d, __doc_hxtorch_conv1d_2, pybind11::arg("input"),
	    pybind11::arg("weight"), pybind11::arg("bias") = c10::optional<torch::Tensor>(),
	    pybind11::arg("stride"), pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = hxtorch::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	m.def(
	    "expanded_conv1d", (single_stride_expanded_conv1d_type) &hxtorch::expanded_conv1d,
	    __doc_hxtorch_expanded_conv1d, pybind11::arg("input"), pybind11::arg("weight"),
	    pybind11::arg("bias") = c10::optional<torch::Tensor>(), pybind11::arg("stride") = 1,
	    pybind11::arg("num_expansions") = 1, pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = hxtorch::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	m.def(
	    "expanded_conv1d", (expanded_conv1d_type) &hxtorch::expanded_conv1d,
	    __doc_hxtorch_expanded_conv1d_2, pybind11::arg("input"), pybind11::arg("weight"),
	    pybind11::arg("bias") = c10::optional<torch::Tensor>(), pybind11::arg("stride"),
	    pybind11::arg("num_expansions") = 1, pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = hxtorch::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	m.def(
	    "conv2d", (single_stride_conv_type) &hxtorch::conv2d, __doc_hxtorch_conv2d,
	    pybind11::arg("input"), pybind11::arg("weight"),
	    pybind11::arg("bias") = c10::optional<torch::Tensor>(), pybind11::arg("stride") = 1,
	    pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = hxtorch::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	m.def(
	    "conv2d", (conv2d_type) &hxtorch::conv2d, __doc_hxtorch_conv2d_2, pybind11::arg("input"),
	    pybind11::arg("weight"), pybind11::arg("bias") = c10::optional<torch::Tensor>(),
	    pybind11::arg("stride"), pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = hxtorch::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	pybind11::class_<hxtorch::MockParameter>(m, "MockParameter", __doc_hxtorch_MockParameter)
	    .def(
	        pybind11::init<double, double>(), __doc_hxtorch_MockParameter_MockParameter,
	        pybind11::arg("noise_std") = hxtorch::constants::defaults::noise_std,
	        pybind11::arg("gain") = hxtorch::constants::defaults::gain)
	    .def_readwrite("noise_std", &hxtorch::MockParameter::noise_std)
	    .def_readwrite("gain", &hxtorch::MockParameter::gain)
	    .def(
	        "__repr__",
	        [](const hxtorch::MockParameter& p) {
		        return "MockParameter(noise_std=" + std::to_string(p.noise_std) +
		               ", gain=" + std::to_string(p.gain) + ")";
	        })
	    .def("__eq__", [](const hxtorch::MockParameter& p1, const hxtorch::MockParameter& p2) {
		    return (p1.gain == p2.gain) && (p1.noise_std == p2.noise_std);
	    });

	pybind11::class_<hxtorch::InferenceTracer>(m, "InferenceTracer", __doc_hxtorch_InferenceTracer)
	    .def(
	        pybind11::init<std::string const&>(), __doc_hxtorch_InferenceTracer_InferenceTracer,
	        pybind11::arg("filename"))
	    .def("stop", &hxtorch::InferenceTracer::stop, __doc_hxtorch_InferenceTracer_stop)
	    .def("start", &hxtorch::InferenceTracer::start, __doc_hxtorch_InferenceTracer_start);

	pybind11::class_<hxtorch::HWDBPath>(m, "HWDBPath", __doc_hxtorch_HWDBPath)
	    .def(
	        pybind11::init<std::optional<std::string>, std::string>(),
	        __doc_hxtorch_HWDBPath_HWDBPath, pybind11::arg("path") = std::nullopt,
	        pybind11::arg("version") = "stable/latest");
	pybind11::class_<hxtorch::CalibrationPath>(m, "CalibrationPath", __doc_hxtorch_CalibrationPath)
	    .def(
	        pybind11::init<std::string>(), __doc_hxtorch_CalibrationPath_CalibrationPath,
	        pybind11::arg("value"));

	auto constants_module = m.def_submodule("constants", "");
	constants_module.attr("synaptic_weight_min") = hxtorch::constants::synaptic_weight_min;
	constants_module.attr("synaptic_weight_max") = hxtorch::constants::synaptic_weight_max;
	constants_module.attr("input_activation_min") = hxtorch::constants::input_activation_min;
	constants_module.attr("input_activation_max") = hxtorch::constants::input_activation_max;
	constants_module.attr("output_activation_min") = hxtorch::constants::output_activation_min;
	constants_module.attr("output_activation_max") = hxtorch::constants::output_activation_max;
	constants_module.attr("hardware_matrix_height") = hxtorch::constants::hardware_matrix_height;
	constants_module.attr("hardware_matrix_width") = hxtorch::constants::hardware_matrix_width;

	auto constants_defaults_module = constants_module.def_submodule("defaults", "");
	constants_defaults_module.attr("wait_between_events") =
	    hxtorch::constants::defaults::wait_between_events;
	constants_defaults_module.attr("gain") = hxtorch::constants::defaults::gain;
	constants_defaults_module.attr("noise_std") = hxtorch::constants::defaults::noise_std;

	auto m_snn = m.def_submodule("_snn");

	pybind11::class_<hxtorch::snn::DataHandle>(m_snn, "DataHandle")
	    .def(pybind11::init<torch::Tensor, float>(), pybind11::arg("data"), pybind11::arg("dt"))
	    .def_property(
	        "data", &hxtorch::snn::DataHandle::get_data, &hxtorch::snn::DataHandle::set_data);
	pybind11::class_<hxtorch::snn::SpikeHandle>(m_snn, "SpikeHandle")
	    .def(pybind11::init<torch::Tensor, float>(), pybind11::arg("data"), pybind11::arg("dt"))
	    .def("get_data", &hxtorch::snn::SpikeHandle::get_data)
	    .def(
	        "set_data", &hxtorch::snn::DataHandle::set_data, pybind11::arg("data"),
	        pybind11::arg("dt"))
	    .def(
	        "to_dense", static_cast<torch::Tensor (hxtorch::snn::SpikeHandle::*)(float)>(
	                        &hxtorch::snn::SpikeHandle::to_dense));
	pybind11::class_<hxtorch::snn::CADCHandle>(m_snn, "CADCHandle")
	    .def(pybind11::init<torch::Tensor, float>(), pybind11::arg("data"), pybind11::arg("dt"))
	    .def("get_data", &hxtorch::snn::CADCHandle::get_data)
	    .def(
	        "set_data", &hxtorch::snn::CADCHandle::set_data, pybind11::arg("data"),
	        pybind11::arg("dt"))
	    .def(
	        "to_dense",
	        static_cast<torch::Tensor (hxtorch::snn::CADCHandle::*)(float, std::string)>(
	            &hxtorch::snn::CADCHandle::to_dense),
	        pybind11::arg("dt"), pybind11::arg("mode") = "linear")
	    .def(
	        "to_dense",
	        static_cast<std::tuple<torch::Tensor, float> (hxtorch::snn::CADCHandle::*)(
	            std::string)>(&hxtorch::snn::CADCHandle::to_dense),
	        pybind11::arg("mode") = "linear");
	pybind11::class_<hxtorch::snn::MADCHandle>(m_snn, "MADCHandle")
	    .def(pybind11::init<torch::Tensor, float>(), pybind11::arg("data"), pybind11::arg("dt"))
	    .def_property(
	        "data", &hxtorch::snn::MADCHandle::get_data, &hxtorch::snn::MADCHandle::set_data);
	m_snn.def(
	    "run", &hxtorch::snn::run, pybind11::arg("config"), pybind11::arg("network_graph"),
	    pybind11::arg("inputs"), pybind11::arg("playback_hooks"));
	m_snn.def("weight_to_connection", &hxtorch::snn::weight_to_connection, pybind11::arg("weight"));

	m_snn.def(
	    "tensor_to_spike_times", &hxtorch::snn::tensor_to_spike_times, pybind11::arg("times"),
	    pybind11::arg("dt"));
	m_snn.def(
	    "extract_spikes", &hxtorch::snn::extract_spikes, pybind11::arg("data"),
	    pybind11::arg("network_graph"), pybind11::arg("runtime"));
	m_snn.def(
	    "extract_cadc", &hxtorch::snn::extract_cadc, pybind11::arg("data"),
	    pybind11::arg("network_graph"), pybind11::arg("runtime"));
	m_snn.def(
	    "extract_madc", &hxtorch::snn::extract_madc, pybind11::arg("data"),
	    pybind11::arg("network_graph"), pybind11::arg("runtime"));
}
