#include <torch/torch.h>

#include <torch/custom_class.h>
#include <torch/script.h>
// for proper handling of torch's python tensor to C++ torch::Tensor conversion
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/extension.h>

#include "hxtorch/connection.h"
#include "hxtorch/conv.h"
#include "hxtorch/detail/conv.h"
#include "hxtorch/mac.h"
#include "hxtorch/matmul.h"
#include "hxtorch/mock.h"
#include "hxtorch/relu.h"

#include "grenade/vx/config.h"
#include "pyhxcomm/vx/connection_handle.h"

namespace hxtorch::detail {

template <typename... Ts>
struct InitUnrollPyBind11Helper
{
	InitUnrollPyBind11Helper(pybind11::module&){};
};

template <typename T, typename... Ts>
struct InitUnrollPyBind11Helper<std::variant<T, Ts...>>
    : InitUnrollPyBind11Helper<std::variant<Ts...>>
{
	using parent_t = InitUnrollPyBind11Helper<std::variant<Ts...>>;

	InitUnrollPyBind11Helper(pybind11::module& m) : parent_t(m)
	{
		m.def(
		    "init",
		    [](grenade::vx::ChipConfig const& chip, T& conn) {
			    hxtorch::init(
			        chip,
			        std::make_unique<hxcomm::vx::ConnectionVariant>(std::move(*conn.release())));
		    },
		    pybind11::arg("chip"), pybind11::arg("connection"));
		m.def(
		    "init",
		    [](std::string const& calibration_path, T& conn) {
			    hxtorch::init(
			        calibration_path,
			        std::make_unique<hxcomm::vx::ConnectionVariant>(std::move(*conn.release())));
		    },
		    pybind11::arg("calibration_path"), pybind11::arg("connection"));
	}
};

} // namespace hxtorch::detail


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	pybind11::module::import("pygrenade_vx");
	pybind11::module::import("pyhxcomm_vx");

	[[maybe_unused]] hxtorch::detail::InitUnrollPyBind11Helper<
	    std::remove_cvref_t<pyhxcomm::vx::ConnectionHandle>>
	    helper(m);
	m.def(
	    "init",
	    [](std::optional<std::string> const& hwdb_path = std::nullopt) {
		    hxtorch::init(hwdb_path);
	    },
	    pybind11::arg("hwdb_path") = std::nullopt);
	m.def("release", &hxtorch::release);
	m.def(
	    "init", (void (*)(hxtorch::MockParameter const&)) & hxtorch::init, "",
	    pybind11::arg("parameter"));
	m.def(
	    "mac", &hxtorch::mac, "", pybind11::arg("x"), pybind11::arg("weights"),
	    pybind11::arg("num_sends") = 1, pybind11::arg("wait_between_events") = 25,
	    pybind11::arg("mock") = false);
	m.def("relu", &hxtorch::relu, "", pybind11::arg("input"), pybind11::arg("mock") = false);
	m.def(
	    "converting_relu", &hxtorch::converting_relu, "", pybind11::arg("input"),
	    pybind11::arg("shift") = 2, pybind11::arg("mock") = false);
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
	    pybind11::arg("wait_between_events") = 25, pybind11::arg("mock") = false);

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
	    pybind11::arg("wait_between_events") = 25, pybind11::arg("mock") = false);
	m.def(
	    "conv1d", (conv1d_type) &hxtorch::conv1d, "", pybind11::arg("input"),
	    pybind11::arg("weight"), pybind11::arg("bias") = c10::optional<torch::Tensor>(),
	    pybind11::arg("stride"), pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = 25, pybind11::arg("mock") = false);
	m.def(
	    "conv2d", (single_stride_conv_type) &hxtorch::conv2d, "", pybind11::arg("input"),
	    pybind11::arg("weight"), pybind11::arg("bias") = c10::optional<torch::Tensor>(),
	    pybind11::arg("stride") = 1, pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = 25, pybind11::arg("mock") = false);
	m.def(
	    "conv2d", (conv2d_type) &hxtorch::conv2d, "", pybind11::arg("input"),
	    pybind11::arg("weight"), pybind11::arg("bias") = c10::optional<torch::Tensor>(),
	    pybind11::arg("stride"), pybind11::arg("num_sends") = 1,
	    pybind11::arg("wait_between_events") = 25, pybind11::arg("mock") = false);
	pybind11::class_<hxtorch::MockParameter>(m, "MockParameter")
	    .def(pybind11::init<>())
	    .def(pybind11::init<float, float>(), pybind11::arg("noise_std"), pybind11::arg("gain"))
	    .def_readwrite("noise_std", &hxtorch::MockParameter::noise_std)
	    .def_readwrite("gain", &hxtorch::MockParameter::gain);
}
