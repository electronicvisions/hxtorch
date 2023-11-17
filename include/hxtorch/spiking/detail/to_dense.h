#include <torch/torch.h>


namespace hxtorch::spiking::detail {

/* Convert sparse spike tensor to a dense representation
 * Spikes are assigned via the nearest neighbor principle
 * @param data Sparse tensor holding the spike events
 * @param sparse_dt The temporal resolution of the sparse tensor
 * @param dt The desired temporal resolution of the dense output tensor
 */
torch::Tensor sparse_spike_to_dense(
    std::vector<std::tuple<int64_t, int64_t, int64_t>> const& data,
    int batch_size,
    int population_size,
    float runtime,
    float dt);


/* Convert sparse MADC tensor to a dense raw-data representation
 * MADC values remain untouched but are shortened along the time dimension, such that batch entries
 * have the same number of MADC samples.
 * @param data Sparse tensor holding the MADC events
 */
torch::Tensor sparse_madc_to_dense_raw(
    std::vector<std::tuple<int16_t, int64_t, int64_t, int64_t>> const& data, int batch_size);


/* Convert sparse CADC tensor to a dense representation
 * CADC values are linearly interpolated along the time dimension
 * @param data Sparse tensor holding the CADC events
 * @param sparse_dt The temporal resolution of the sparse tensor
 * @param dt The desired temporal resolution of the dense output tensor
 */
torch::Tensor sparse_cadc_to_dense_linear(
    std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>> const& data,
    int batch_size,
    int population_size,
    float runtime,
    float dt);


/* Convert sparse CADC tensor to a dense representation
 * CADC values are nearest-neighbor interpolated along the time dimension
 * @param data Sparse tensor holding the CADC events
 * @param sparse_dt The temporal resolution of the sparse tensor
 * @param dt The desired temporal resolution of the dense output tensor
 */
torch::Tensor sparse_cadc_to_dense_nn(
    std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>> const& data,
    int batch_size,
    int population_size,
    float runtime,
    float dt);

/* Convert sparse CADC tensor to a dense raw-data representation
 * CADC values remain untouched but are shortened along the time dimension, such that all neurons
 * have the same number of CADC samples.
 * @param data Sparse tensor holding the CADC events
 */
std::tuple<torch::Tensor, torch::Tensor> sparse_cadc_to_dense_raw(
    std::vector<std::tuple<int32_t, int64_t, int64_t, int64_t>> const& data,
    int batch_size,
    int population_size);

} // namespace hxtorch::spiking::detail
