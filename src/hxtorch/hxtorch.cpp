#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <torch/custom_class.h>
#include <torch/script.h>
// for proper handling of torch's python tensor to C++ torch::Tensor conversion
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/extension.h>

#include "grenade/vx/compute_single_mac.h"
#include "grenade/vx/config.h"
#include "pyhxcomm/vx/connection_handle.h"
#include "stadls/vx/init_generator.h"

std::unique_ptr<hxcomm::vx::ConnectionVariant>& getConnection()
{
	static std::unique_ptr<hxcomm::vx::ConnectionVariant> connection;
	return connection;
}

static grenade::vx::ChipConfig& getChip()
{
	static grenade::vx::ChipConfig chip;
	return chip;
}

void init(
    grenade::vx::ChipConfig const& chip, std::unique_ptr<hxcomm::vx::ConnectionVariant> connection)
{
	getChip() = chip;
	getConnection() = std::move(connection);
}

void release()
{
	getConnection().reset();
}

torch::Tensor mac_forward(
    torch::Tensor x, torch::Tensor weights, int64_t num_sends, int64_t wait_between_events)
{
	if (x.dim() != 1) {
		throw std::runtime_error("HICANN-X only supports 1D input");
	}

	if (!x.is_contiguous()) {
		throw std::runtime_error("HICANN-X only supports contiguous input vectors");
	}

	if (weights.dim() != 2) {
		throw std::runtime_error("HICANN-X only supports 2D weight matrices");
	}

	if (!weights.is_contiguous()) {
		throw std::runtime_error("HICANN-X only supports contiguous weight matrices");
	}

	// create vector
	size_t const num_double_rows = weights.sizes().vec().at(0);
	size_t const num_rows = 2 * num_double_rows;
	size_t const num_cols = weights.sizes().vec().at(1);

	if (static_cast<size_t>(x.sizes().vec().at(0)) != num_double_rows) {
		throw std::runtime_error("HICANN-X only supports vector lengths that match the "
		                         "corresponding weight matrix dim size");
	}

	std::vector<std::vector<haldls::vx::SynapseQuad::Weight>> m_weights{
	    num_rows, std::vector<haldls::vx::SynapseQuad::Weight>{num_cols}};

	// TODO: let's assume it's floats...
	auto weights_a = weights.accessor<float, 2>();
	for (size_t i = 0; i < num_double_rows; i++) {
		for (size_t j = 0; j < num_cols; j++) {
			// FIXME: some float to int conversion here
			size_t row = 2 * i + (weights_a[i][j] < 0);
			m_weights[row][j] = haldls::vx::SynapseQuad::Weight(std::abs(weights_a[i][j]));
		}
	}

	std::vector<grenade::vx::UInt5> xin(num_rows);
	std::vector<haldls::vx::SynapseDriverConfig::RowMode> row_modes(num_rows);

	// TODO: let's assume it's floats...
	auto x_a = x.accessor<float, 1>();
	for (size_t i = 0; i < num_double_rows; i++) {
		// FIXME: some float to int conversion here
		xin[2 * i] = grenade::vx::UInt5(x_a[i]);
		xin[2 * i + 1] = grenade::vx::UInt5(x_a[i]);
		row_modes[2 * i] = haldls::vx::SynapseDriverConfig::RowMode::excitatory;
		row_modes[2 * i + 1] = haldls::vx::SynapseDriverConfig::RowMode::inhibitory;
	}

	grenade::vx::ComputeSingleMAC mac{m_weights, row_modes, getChip(),
	                                  static_cast<size_t>(num_sends),
	                                  haldls::vx::Timer::Value(wait_between_events)};

	if (!getConnection()) {
		throw std::runtime_error("No connection allocated.");
	}
	auto const results = mac.run({xin}, *getConnection()).at(0);

	torch::Tensor ret = torch::zeros({static_cast<int>(results.size())});
	auto ret_a = ret.accessor<float, 1>();
	for (size_t i = 0; i < results.size(); i++) {
		ret_a[i] = results[i];
	}

	return ret;
}

torch::autograd::variable_list mac_backward(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor weights)
{
	// TODO: does this represent roughly our MAC?
	auto grad_x = grad_output.matmul(weights.t());
	if (x.dim() == 1) {
		x = x.reshape({1, -1});
	}
	if (grad_output.dim() == 1) {
		grad_output = grad_output.reshape({1, -1});
	}
	auto grad_weights = x.t().matmul(grad_output);
	return {grad_x, grad_weights, {}, {}};
}


class MAC : public torch::autograd::Function<MAC>
{
public:
	static torch::autograd::variable_list forward(
	    torch::autograd::AutogradContext* ctx,
	    torch::autograd::Variable x,
	    torch::autograd::Variable weights,
	    int64_t num_sends,
	    int64_t wait_between_events)
	{
		ctx->save_for_backward({x, weights});
		auto ret = mac_forward(x, weights, num_sends, wait_between_events);
		return {ret};
	}

	static torch::autograd::variable_list backward(
	    torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output)
	{
		auto saved_variables = ctx->get_saved_variables();
		auto x = saved_variables[0];
		auto weights = saved_variables[1];
		return mac_backward(grad_output[0], x, weights);
	}
};

torch::Tensor mac(
    torch::Tensor const& x,
    torch::Tensor const& weights,
    int64_t const num_sends = 1,
    int64_t const wait_between_events = 25)
{
	auto ret = MAC::apply(x, weights, num_sends, wait_between_events);
	return ret[0];
}

static auto registry = torch::RegisterOperators().op("hxtorch::mac", &mac);


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
			init(chip, std::make_unique<hxcomm::vx::ConnectionVariant>(std::move(*conn.release())));
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
	m.def("release", &release);
	m.def(
	    "mac", &mac, "", pybind11::arg("x"), pybind11::arg("weights"),
	    pybind11::arg("num_sends") = 1, pybind11::arg("wait_between_events") = 25);
}
