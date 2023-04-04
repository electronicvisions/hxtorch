/*
 * This file contains docstrings for use in the Python bindings.
 * FIXME: the docstrings have to be manually syncronized with the corresponding headers
 */

static const char* __doc_hxtorch_CalibrationPath = R"doc(Path to a calibration.)doc";

static const char* __doc_hxtorch_CalibrationPath_CalibrationPath = R"doc()doc";

static const char* __doc_hxtorch_HWDBPath = R"doc(Path to a hardware database.)doc";

static const char* __doc_hxtorch_HWDBPath_HWDBPath = R"doc()doc";

static const char* __doc_hxtorch_init_hardware =
    R"doc(Initialize the hardware automatically from the environment.

@param hwdb_path Optional path to the hwdb to use
@param spiking Boolean flag indicating whether spiking or non-spiking calibration is loaded)doc";

static const char* __doc_hxtorch_init_hardware_2 =
    R"doc(Initialize the hardware with calibration path.

@param calibration_path Calibration path to load from)doc";

static const char* __doc_hxtorch_init_hardware_minimal =
    R"doc(Initialize automatically from the environment
without ExperimentInit and without any calibration.)doc";

static const char* __doc_hxtorch_get_unique_identifier =
    R"doc(Return the unique identifier of the chip with the initialized connection.

@param hwdb_path Optional path to the hwdb to use
@return The identifier as string)doc";

static const char* __doc_hxtorch_release_hardware = R"doc(Release hardware resource.)doc";
