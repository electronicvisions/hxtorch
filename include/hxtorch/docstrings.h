/*
 * This file contains docstrings for use in the Python bindings.
 * FIXME: the docstrings have to be manually syncronized with the corresponding headers
 */

static const char* __doc_hxtorch_CalibrationPath = R"doc(Path to a calibration.)doc";

static const char* __doc_hxtorch_CalibrationPath_CalibrationPath = R"doc()doc";

static const char* __doc_hxtorch_HWDBPath = R"doc(Path to a hardware database.)doc";

static const char* __doc_hxtorch_HWDBPath_HWDBPath = R"doc()doc";

static const char* __doc_hxtorch_InferenceTracer =
    R"doc(Inference tracer for a linear sequence of operations.
The traced operations' state is saved as a grenade::compute::Sequence,
which can be executed as a single operation without transformation to and
from PyTorch tensors.
It is ensured, that no untraced modifications are made in-between traced
operations by comparing the last traced operation's output with the
currently traced operation's input value.

@note
Not final API or implementation, see Issue #3694)doc";

static const char* __doc_hxtorch_InferenceTracer_InferenceTracer =
    R"doc(Construct inference tracer with filename to store traced operations to.)doc";

static const char* __doc_hxtorch_InferenceTracer_start =
    R"doc(Start tracing operations by registering tracer.)doc";

static const char* __doc_hxtorch_InferenceTracer_stop =
    R"doc(Stop tracing operations by deregistering tracer and save traced
operations to given file.

@return List of traced operation names)doc";

static const char* __doc_hxtorch_MockParameter = R"doc(Parameter of hardware mock.)doc";

static const char* __doc_hxtorch_MockParameter_MockParameter =
    R"doc(Construct with noise standard deviation and gain.

@param noise_std Noise standard deviation to use
@param gain Gain to use)doc";

static const char* __doc_hxtorch_add =
    R"doc(Elementwise addition operating on int8 value range.

@param input Input tensor
@param other Other tensor, which must be broadcastable to input tensor dimension
@param alpha The scalar multiplier for other
@param mock Enable mock mode)doc";

static const char* __doc_hxtorch_argmax =
    R"doc(Arg max operation on int8 value range.

@param input The input tensor
@param dim The dimension to reduce. If unspecified, the argmax of the flattened
           input is returned.
@param keepdim Whether the output tensor has @p dim retained or not. Ignored
               if @p dim is unspecified.
@param mock Enable mock mode

@return The indices of the maximum values of a tensor across a dimension)doc";

static const char* __doc_hxtorch_conv1d = R"doc()doc";

static const char* __doc_hxtorch_conv1d_2 = R"doc()doc";

static const char* __doc_hxtorch_conv2d = R"doc()doc";

static const char* __doc_hxtorch_conv2d_2 = R"doc()doc";

static const char* __doc_hxtorch_converting_relu =
    R"doc(Rectified linear unit operating on int8 value range converting to uint5
value range.
The result is bit-shifted by @p shift after applying the ReLU and clipped
to the input range of BrainScaleS-2.

@param input Input tensor
@param shift Amount of bits to shift before clipping
@param mock Enable mock mode)doc";

static const char* __doc_hxtorch_get_mock_parameter =
    R"doc(Returns the current mock parameters.)doc";

static const char* __doc_hxtorch_inference_trace =
    R"doc(Execute inference of stored trace.

@param input Input data to use
@param filename Filename to serialized operation trace)doc";

static const char* __doc_hxtorch_init_hardware =
    R"doc(Initialize the hardware automatically from the environment.

@param calibration_version Calibration version to load
@param hwdb_path Optional path to the hwdb to use)doc";

static const char* __doc_hxtorch_init_hardware_2 =
    R"doc(Initialize the hardware with calibration path.

@param calibration_path Calibration path to load from)doc";

static const char* __doc_hxtorch_init_hardware_minimal =
    R"doc(Initialize automatically from the environment
without ExperimentInit and without any calibration.)doc";

static const char* __doc_hxtorch_mac =
    R"doc(The bare mutliply-accumulate operation of BrainScaleS-2. A 1D input @p x
is multiplied by the weight matrix @p weights. If @p x is two-dimensional,
the weights are sent only once to the synapse array and the inputs are
consecutively multiplied as a 1D vector.

@param x Input tensor
@param weights The weights of the synapse array
@param num_sends How often to send the (same) input vector
@param wait_between_events How long to wait (in FPGA cycles) between events
@param mock Enable mock mode

@return Resulting tensor)doc";

static const char* __doc_hxtorch_matmul =
    R"doc(Drop-in replacement for the torch.matmul operation that uses BrainScaleS-2.

@note
The current implementation only supports @p other to be 1D or 2D.

@param input First input tensor
@param other Second input tensor
@param num_sends How often to send the (same) input vector
@param wait_between_events How long to wait (in FPGA cycles) between events
@param mock: Enable mock mode

@return Resulting tensor)doc";

static const char* __doc_hxtorch_measure_mock_parameter =
    R"doc(Measures the mock parameters, i.e. gain and noise_std, by multiplying a
full weight with an artificial test input on the BSS-2 chip.
For this purpose a random pattern is used, whose mean value is successively
reduced to also work with higher gain factors.
The output for the actual calibration is chosen such that it is close to
the middle of the available range.)doc";

static const char* __doc_hxtorch_release_hardware = R"doc(Release hardware resource.)doc";

static const char* __doc_hxtorch_relu =
    R"doc(Rectified linear unit operating on int8 value range.

@param input Input tensor
@param mock Enable mock mode)doc";

static const char* __doc_hxtorch_set_mock_parameter = R"doc(Sets the mock parameters.)doc";
