#include <torch/torch.h>

#include <torch/custom_class.h>
#include <torch/script.h>
// for proper handling of torch's python tensor to C++ torch::Tensor conversion
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/extension.h>

#include "hxtorch/connection.h"
#include "hxtorch/mac.h"
#include "hxtorch/matmul.h"

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
		m.def("init", [](grenade::vx::ChipConfig const& chip, T& conn) {
			hxtorch::init(
			    chip, std::make_unique<hxcomm::vx::ConnectionVariant>(std::move(*conn.release())));
		});
		m.def("init", [](std::string const& calibration_path, T& conn) {
			hxtorch::init(
			    calibration_path,
			    std::make_unique<hxcomm::vx::ConnectionVariant>(std::move(*conn.release())));
		});
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
	    "mac", &hxtorch::mac, "", pybind11::arg("x"), pybind11::arg("weights"),
	    pybind11::arg("num_sends") = 1, pybind11::arg("wait_between_events") = 25);
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
	    pybind11::arg("wait_between_events") = 25);
}
