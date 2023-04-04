#include <type_traits>
#include <vector>
#include <torch/torch.h>

namespace hxtorch::perceptron {

namespace detail {

/** Implementation for the
 *      at::TensorAccessor<T, N, PtrTraits, index_t>
 *  to
 *      std::vector<...<R>> (nesting-level N)
 *  conversion helper below.
 *
 *  @tparam R return value type
 *  @tparam Trafo transformation function
 *  @note Other template parameaters come from at::TensorAccessor
 */
template <
    typename R,
    typename Trafo,
    typename T,
    size_t N,
    template <typename U>
    class PtrTraits,
    typename index_t>
struct ConvertToVector
{
private:
	typedef ConvertToVector<R, Trafo, T, N - 1, PtrTraits, index_t> unpacked_type;

public:
	typedef std::vector<typename unpacked_type::result_type> result_type;
	typedef at::TensorAccessor<T, N, PtrTraits, index_t> value_type;

	static result_type apply(value_type const& value, Trafo t)
	{
		result_type result;
		result.reserve(value.size(0));

		for (index_t i = 0; i < value.size(0); i++) {
			result.push_back(unpacked_type::apply(value[i], t));
		}

		return result;
	}
};

/* special case for N=1 which finally provides value_type access via operator[] */
template <
    typename R,
    typename Trafo,
    typename T,
    template <typename U>
    class PtrTraits,
    typename index_t>
struct ConvertToVector<R, Trafo, T, 1, PtrTraits, index_t>
{
	typedef std::vector<R> result_type;
	typedef at::TensorAccessor<T, 1, PtrTraits, index_t> value_type;

	static result_type apply(value_type const& value, Trafo t)
	{
		result_type result;
		result.reserve(value.size(0));

		for (index_t i = 0; i < value.size(0); i++) {
			result.push_back(t(value[i]));
		}

		return result;
	}
};

/** Default transformation function from T to R. */
template <typename R, typename T>
R default_transform(T const& t)
{
	return static_cast<R>(t);
}

} // namespace detail

/** Conversion helper for converting
 *      at::TensorAccessor<T, N, PtrTraits, index_t>
 *  to
 *      std::vector<...<R>> (nesting-level N)
 *  types.
 *
 *  @tparam R value_type to be returned
 *  @tparam Trafo Conversion type (defaults to R)
 *  @param tensor The tensor accessor to be converted to a nested vector
 *  @param func The conversion function for the value_type inside (defaults to R())
 *  @note The underlying value_type T is converted to R. All other template parameters come from
 * at::TensorAccessor
 *  @return A nested std::vector<std::vector<...<R>>> (N nested vectors).
 */
template <
    typename R,
    typename T,
    size_t N,
    template <typename U>
    class PtrTraits,
    typename index_t,
    typename Trafo = decltype(detail::default_transform<R, T>)>
typename detail::ConvertToVector<R, Trafo, T, N, PtrTraits, index_t>::result_type convert_to_vector(
    at::TensorAccessor<T, N, PtrTraits, index_t> const& tensor,
    Trafo func = detail::default_transform)
{
	return detail::ConvertToVector<R, Trafo, T, N, PtrTraits, index_t>::apply(tensor, func);
}

} // namespace hxtorch::perceptron
