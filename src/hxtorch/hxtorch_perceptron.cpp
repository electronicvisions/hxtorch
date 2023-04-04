#include <torch/torch.h>

#include <torch/custom_class.h>
#include <torch/script.h>
// for proper handling of torch's python tensor to C++ torch::Tensor conversion
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/extension.h>

#include "hxtorch/perceptron/add.h"
#include "hxtorch/perceptron/argmax.h"
#include "hxtorch/perceptron/constants.h"
#include "hxtorch/perceptron/conv.h"
#include "hxtorch/perceptron/detail/conv.h"
#include "hxtorch/perceptron/detail/mock.h"
#include "hxtorch/perceptron/docstrings.h"
#include "hxtorch/perceptron/inference_tracer.h"
#include "hxtorch/perceptron/mac.h"
#include "hxtorch/perceptron/matmul.h"
#include "hxtorch/perceptron/mock.h"
#include "hxtorch/perceptron/relu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.import("pygrenade_vx");
	m.def(
	    "get_mock_parameter", &hxtorch::perceptron::get_mock_parameter,
	    __doc_hxtorch_get_mock_parameter);
	m.def(
	    "set_mock_parameter", &hxtorch::perceptron::set_mock_parameter,
	    __doc_hxtorch_set_mock_parameter, pybind11::arg("parameter"));
	m.def(
	    "mac", &hxtorch::perceptron::mac, __doc_hxtorch_mac, pybind11::arg("x"),
	    pybind11::arg("weights"), pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") =
	        hxtorch::perceptron::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	m.def(
	    "measure_mock_parameter", &hxtorch::perceptron::measure_mock_parameter,
	    __doc_hxtorch_measure_mock_parameter);
	m.def(
	    "relu", &hxtorch::perceptron::relu, __doc_hxtorch_relu, pybind11::arg("input"),
	    pybind11::arg("mock") = false);
	m.def(
	    "converting_relu", &hxtorch::perceptron::converting_relu, __doc_hxtorch_converting_relu,
	    pybind11::arg("input"), pybind11::arg("shift") = 2, pybind11::arg("mock") = false);
	m.def(
	    "inference_trace", &hxtorch::perceptron::inference_trace, __doc_hxtorch_inference_trace,
	    pybind11::arg("input"), pybind11::arg("filename"));
	m.def(
	    "argmax", &hxtorch::perceptron::argmax, __doc_hxtorch_argmax, pybind11::arg("input"),
	    pybind11::arg("dim") = c10::optional<int64_t>(), pybind11::arg("keepdim") = false,
	    pybind11::arg("mock") = false);
	m.def(
	    "add", &hxtorch::perceptron::add, __doc_hxtorch_add, pybind11::arg("input"),
	    pybind11::arg("other"), pybind11::arg("alpha") = 1, pybind11::arg("mock") = false);
	m.def(
	    "matmul", &hxtorch::perceptron::matmul, __doc_hxtorch_matmul, pybind11::arg("input"),
	    pybind11::arg("other"), pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") =
	        hxtorch::perceptron::constants::defaults::wait_between_events,
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
	    "conv1d", (single_stride_conv_type) &hxtorch::perceptron::conv1d, __doc_hxtorch_conv1d,
	    pybind11::arg("input"), pybind11::arg("weight"),
	    pybind11::arg("bias") = c10::optional<torch::Tensor>(), pybind11::arg("stride") = 1,
	    pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") =
	        hxtorch::perceptron::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	m.def(
	    "conv1d", (conv1d_type) &hxtorch::perceptron::conv1d, __doc_hxtorch_conv1d_2,
	    pybind11::arg("input"), pybind11::arg("weight"),
	    pybind11::arg("bias") = c10::optional<torch::Tensor>(), pybind11::arg("stride"),
	    pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") =
	        hxtorch::perceptron::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	m.def(
	    "expanded_conv1d",
	    (single_stride_expanded_conv1d_type) &hxtorch::perceptron::expanded_conv1d,
	    __doc_hxtorch_expanded_conv1d, pybind11::arg("input"), pybind11::arg("weight"),
	    pybind11::arg("bias") = c10::optional<torch::Tensor>(), pybind11::arg("stride") = 1,
	    pybind11::arg("num_expansions") = 1, pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") =
	        hxtorch::perceptron::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	m.def(
	    "expanded_conv1d", (expanded_conv1d_type) &hxtorch::perceptron::expanded_conv1d,
	    __doc_hxtorch_expanded_conv1d_2, pybind11::arg("input"), pybind11::arg("weight"),
	    pybind11::arg("bias") = c10::optional<torch::Tensor>(), pybind11::arg("stride"),
	    pybind11::arg("num_expansions") = 1, pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") =
	        hxtorch::perceptron::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	m.def(
	    "conv2d", (single_stride_conv_type) &hxtorch::perceptron::conv2d, __doc_hxtorch_conv2d,
	    pybind11::arg("input"), pybind11::arg("weight"),
	    pybind11::arg("bias") = c10::optional<torch::Tensor>(), pybind11::arg("stride") = 1,
	    pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") =
	        hxtorch::perceptron::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	m.def(
	    "conv2d", (conv2d_type) &hxtorch::perceptron::conv2d, __doc_hxtorch_conv2d_2,
	    pybind11::arg("input"), pybind11::arg("weight"),
	    pybind11::arg("bias") = c10::optional<torch::Tensor>(), pybind11::arg("stride"),
	    pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") =
	        hxtorch::perceptron::constants::defaults::wait_between_events,
	    pybind11::arg("mock") = false);
	pybind11::class_<hxtorch::perceptron::MockParameter>(
	    m, "MockParameter", __doc_hxtorch_MockParameter)
	    .def(
	        pybind11::init<double, double>(), __doc_hxtorch_MockParameter_MockParameter,
	        pybind11::arg("noise_std") = hxtorch::perceptron::constants::defaults::noise_std,
	        pybind11::arg("gain") = hxtorch::perceptron::constants::defaults::gain)
	    .def_readwrite("noise_std", &hxtorch::perceptron::MockParameter::noise_std)
	    .def_readwrite("gain", &hxtorch::perceptron::MockParameter::gain)
	    .def(
	        "__repr__",
	        [](const hxtorch::perceptron::MockParameter& p) {
		        return "MockParameter(noise_std=" + std::to_string(p.noise_std) +
		               ", gain=" + std::to_string(p.gain) + ")";
	        })
	    .def(
	        "__eq__", [](const hxtorch::perceptron::MockParameter& p1,
	                     const hxtorch::perceptron::MockParameter& p2) {
		        return (p1.gain == p2.gain) && (p1.noise_std == p2.noise_std);
	        });

	pybind11::class_<hxtorch::perceptron::InferenceTracer>(
	    m, "InferenceTracer", __doc_hxtorch_InferenceTracer)
	    .def(
	        pybind11::init<std::string const&>(), __doc_hxtorch_InferenceTracer_InferenceTracer,
	        pybind11::arg("filename"))
	    .def(
	        "stop", &hxtorch::perceptron::InferenceTracer::stop, __doc_hxtorch_InferenceTracer_stop)
	    .def(
	        "start", &hxtorch::perceptron::InferenceTracer::start,
	        __doc_hxtorch_InferenceTracer_start);

	auto constants_module = m.def_submodule("constants", "");
	constants_module.attr("synaptic_weight_min") =
	    hxtorch::perceptron::constants::synaptic_weight_min;
	constants_module.attr("synaptic_weight_max") =
	    hxtorch::perceptron::constants::synaptic_weight_max;
	constants_module.attr("input_activation_min") =
	    hxtorch::perceptron::constants::input_activation_min;
	constants_module.attr("input_activation_max") =
	    hxtorch::perceptron::constants::input_activation_max;
	constants_module.attr("output_activation_min") =
	    hxtorch::perceptron::constants::output_activation_min;
	constants_module.attr("output_activation_max") =
	    hxtorch::perceptron::constants::output_activation_max;
	constants_module.attr("hardware_matrix_height") =
	    hxtorch::perceptron::constants::hardware_matrix_height;
	constants_module.attr("hardware_matrix_width") =
	    hxtorch::perceptron::constants::hardware_matrix_width;

	auto constants_defaults_module = constants_module.def_submodule("defaults", "");
	constants_defaults_module.attr("wait_between_events") =
	    hxtorch::perceptron::constants::defaults::wait_between_events;
	constants_defaults_module.attr("gain") = hxtorch::perceptron::constants::defaults::gain;
	constants_defaults_module.attr("noise_std") =
	    hxtorch::perceptron::constants::defaults::noise_std;
}
