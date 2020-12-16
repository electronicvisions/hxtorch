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
#include "hxtorch/conv.h"
#include "hxtorch/detail/conv.h"
#include "hxtorch/detail/mock.h"
#include "hxtorch/inference_tracer.h"
#include "hxtorch/mac.h"
#include "hxtorch/matmul.h"
#include "hxtorch/mock.h"
#include "hxtorch/relu.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def(
	    "init", (void (*)(std::optional<hxtorch::HWDBPath> const&)) & hxtorch::init,
	    pybind11::arg("hwdb_path") = std::nullopt);
	m.def(
	    "init", (void (*)(hxtorch::CalibrationPath const&)) & hxtorch::init,
	    pybind11::arg("calibration_path"));
	m.def("release", &hxtorch::release);
	m.def(
	    "init", (void (*)(hxtorch::MockParameter const&)) & hxtorch::init, "",
	    pybind11::arg("parameter"));
	m.def(
	    "mac", &hxtorch::mac, "", pybind11::arg("x"), pybind11::arg("weights"),
	    pybind11::arg("num_sends") = 1, pybind11::arg("wait_between_events") = 5,
	    pybind11::arg("mock") = false);
	m.def("relu", &hxtorch::relu, "", pybind11::arg("input"), pybind11::arg("mock") = false);
	m.def(
	    "converting_relu", &hxtorch::converting_relu, "", pybind11::arg("input"),
	    pybind11::arg("shift") = 2, pybind11::arg("mock") = false);
	m.def(
	    "inference_trace", &hxtorch::inference_trace, "", pybind11::arg("input"),
	    pybind11::arg("filename"));
	m.def(
	    "argmax", &hxtorch::argmax, "", pybind11::arg("input"),
	    pybind11::arg("dim") = c10::optional<int64_t>(), pybind11::arg("keepdim") = false,
	    pybind11::arg("mock") = false);
	m.def(
	    "add", &hxtorch::add, "", pybind11::arg("input"), pybind11::arg("other"),
	    pybind11::arg("alpha") = 1, pybind11::arg("mock") = false);
	m.def(
	    "matmul", &hxtorch::matmul,
	    "Drop-in replacement for :meth:`torch.matmul` that uses HICANN-X.\n"
	    "The current implementation only supports ``other`` to be 1D or 2D.\n\n"
	    ":param input: First input tensor, allowed range [0, 31]\n"
	    ":param other: Second input tensor, allowed range: [-63, 63]\n"
	    ":param num_sends: How often to send the (same) input vector\n"
	    ":param wait_between_events: How long to wait (in FPGA cycles) between events\n"
	    ":returns: Resulting tensor\n",
	    pybind11::arg("input"), pybind11::arg("other"), pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = 5, pybind11::arg("mock") = false);

	typedef torch::Tensor (*single_stride_conv_type)(
	    torch::Tensor const&, torch::Tensor const&, c10::optional<torch::Tensor> const&, int64_t,
	    int64_t, int64_t, bool);
	typedef torch::Tensor (*conv1d_type)(
	    torch::Tensor const&, torch::Tensor const&, c10::optional<torch::Tensor> const&,
	    std::array<int64_t, 1>, int64_t, int64_t, bool);
	typedef torch::Tensor (*conv2d_type)(
	    torch::Tensor const&, torch::Tensor const&, c10::optional<torch::Tensor> const&,
	    std::array<int64_t, 2>, int64_t, int64_t, bool);

	m.def(
	    "conv1d", (single_stride_conv_type) &hxtorch::conv1d, "", pybind11::arg("input"),
	    pybind11::arg("weight"), pybind11::arg("bias") = c10::optional<torch::Tensor>(),
	    pybind11::arg("stride") = 1, pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = 5, pybind11::arg("mock") = false);
	m.def(
	    "conv1d", (conv1d_type) &hxtorch::conv1d, "", pybind11::arg("input"),
	    pybind11::arg("weight"), pybind11::arg("bias") = c10::optional<torch::Tensor>(),
	    pybind11::arg("stride"), pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = 5, pybind11::arg("mock") = false);
	m.def(
	    "conv2d", (single_stride_conv_type) &hxtorch::conv2d, "", pybind11::arg("input"),
	    pybind11::arg("weight"), pybind11::arg("bias") = c10::optional<torch::Tensor>(),
	    pybind11::arg("stride") = 1, pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = 5, pybind11::arg("mock") = false);
	m.def(
	    "conv2d", (conv2d_type) &hxtorch::conv2d, "", pybind11::arg("input"),
	    pybind11::arg("weight"), pybind11::arg("bias") = c10::optional<torch::Tensor>(),
	    pybind11::arg("stride"), pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = 5, pybind11::arg("mock") = false);
	pybind11::class_<hxtorch::MockParameter>(m, "MockParameter")
	    .def(pybind11::init<>())
	    .def(pybind11::init<float, float>(), pybind11::arg("noise_std"), pybind11::arg("gain"))
	    .def_readwrite("noise_std", &hxtorch::MockParameter::noise_std)
	    .def_readwrite("gain", &hxtorch::MockParameter::gain);

	pybind11::class_<hxtorch::InferenceTracer>(m, "InferenceTracer")
	    .def(pybind11::init<std::string const&>(), pybind11::arg("filename"))
	    .def("stop", &hxtorch::InferenceTracer::stop)
	    .def("start", &hxtorch::InferenceTracer::start);

	pybind11::class_<hxtorch::HWDBPath>(m, "HWDBPath")
	    .def(
	        pybind11::init<std::optional<std::string>, std::string>(),
	        pybind11::arg("path") = std::nullopt, pybind11::arg("version") = "stable/latest");
	pybind11::class_<hxtorch::CalibrationPath>(m, "CalibrationPath")
	    .def(pybind11::init<std::string>());
}
