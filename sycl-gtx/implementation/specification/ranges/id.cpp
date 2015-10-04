#include "id.h"
#include "../../data_ref.h"

using namespace cl::sycl;
using detail::id_;
using detail::id_ref;
using detail::data_ref;

template struct id_<1>;
template struct id_<2>;
template struct id_<3>;

template <int dimensions>
id_<dimensions>::id_(size_t first, size_t second, size_t third)
	: type(data_ref::type_t::id_global)
#if MSVC_LOW
{
	values[0] = first;
	values[1] = second;
	values[2] = third;
}
#else
	, dims{ first, second, third } {}
#endif

template <int dimensions>
id_ref id_<dimensions>::operator[](size_t n) {
	return id_ref(n, &values[n], type);
}

#define SYCL_ID_OPERATOR(op)								\
	template <int dimensions>								\
	data_ref id_<dimensions>::operator op(size_t n) const {	\
		return id_ref(0, nullptr, type) op n;				\
	}

SYCL_ID_OPERATOR(+)
SYCL_ID_OPERATOR(-)
SYCL_ID_OPERATOR(*)
SYCL_ID_OPERATOR(/)
SYCL_ID_OPERATOR(%)

SYCL_ID_OPERATOR(>)
SYCL_ID_OPERATOR(<)
SYCL_ID_OPERATOR(>=)
SYCL_ID_OPERATOR(<=)
SYCL_ID_OPERATOR(==)
SYCL_ID_OPERATOR(!=)

#undef SYCL_ID_OPERATOR
