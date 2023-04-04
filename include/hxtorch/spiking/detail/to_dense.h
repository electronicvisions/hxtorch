#include <torch/torch.h>


namespace hxtorch::spiking::detail {

/* Convert sparse spike tensor to a dense representation
 * Spikes are assigned via the nearest neighbor prinzip
 * @param data Sparse tensor holding the spike events
 * @param sparse_dt The temporal resolution of the sparse tensor
 * @param dt The desired temporal resolution of the dense output tensor
 */
torch::Tensor sparse_spike_to_dense(torch::Tensor const& data, float sparse_dt, float dt);


/* Convert sparse CADC tensor to a dense representation
 * CADC values are linearly interpolated along the time dimension
 * @param data Sparse tensor holding the CADC events
 * @param sparse_dt The temporal resolution of the sparse tensor
 * @param dt The desired temporal resolution of the dense output tensor
 */
torch::Tensor sparse_cadc_to_dense_linear(torch::Tensor const& data, float sparse_dt, float dt);


/* Convert sparse CADC tensor to a dense representation
 * CADC values are nearest-neighbor interpolated along the time dimension
 * @param data Sparse tensor holding the CADC events
 * @param sparse_dt The temporal resolution of the sparse tensor
 * @param dt The desired temporal resolution of the dense output tensor
 */
torch::Tensor sparse_cadc_to_dense_nn(torch::Tensor const& data, float sparse_dt, float dt);

/* Convert sparse CADC tensor to a dense raw-data representation
 * CADC values remain untouched but are shortened along the time dimension, such that all neurons
 * have the same number of CADC samples.
 * @param data Sparse tensor holding the CADC events
 */
std::tuple<torch::Tensor, torch::Tensor> sparse_cadc_to_dense_raw(torch::Tensor const& data);

} // namespace hxtorch::spiking::detail
