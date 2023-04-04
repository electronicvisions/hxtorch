#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hxtorch/core/connection.h"
#include "hxtorch/core/docstrings.h"


PYBIND11_MODULE(_hxtorch_core, m)
{
	m.import("pygrenade_vx");
	m.def(
	    "init_hardware",
	    (void (*)(std::optional<hxtorch::core::HWDBPath> const&, bool)) &
	        hxtorch::core::init_hardware,
	    __doc_hxtorch_init_hardware, pybind11::arg("hwdb_path") = std::nullopt,
	    pybind11::arg("ann") = false);
	m.def(
	    "init_hardware",
	    (void (*)(hxtorch::core::CalibrationPath const&)) & hxtorch::core::init_hardware,
	    __doc_hxtorch_init_hardware_2, pybind11::arg("calibration_path"));
	m.def(
	    "init_hardware_minimal", &hxtorch::core::init_hardware_minimal,
	    __doc_hxtorch_init_hardware_minimal);
	m.def(
	    "get_unique_identifier",
	    (std::string(*)(std::optional<hxtorch::core::HWDBPath> const&)) &
	        hxtorch::core::get_unique_identifier,
	    __doc_hxtorch_get_unique_identifier, pybind11::arg("hwdb_path") = std::nullopt);
	m.def("release_hardware", &hxtorch::core::release_hardware, __doc_hxtorch_release_hardware);

	pybind11::class_<hxtorch::core::HWDBPath>(m, "HWDBPath", __doc_hxtorch_HWDBPath)
	    .def(
	        pybind11::init<std::optional<std::string>, std::string>(),
	        __doc_hxtorch_HWDBPath_HWDBPath, pybind11::arg("path") = std::nullopt,
	        pybind11::arg("version") = "stable/latest");
	pybind11::class_<hxtorch::core::CalibrationPath>(
	    m, "CalibrationPath", __doc_hxtorch_CalibrationPath)
	    .def(
	        pybind11::init<std::string>(), __doc_hxtorch_CalibrationPath_CalibrationPath,
	        pybind11::arg("value"));
}
