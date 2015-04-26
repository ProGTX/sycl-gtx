#include "id.h"
#include "../../data_ref.h"

using namespace cl::sycl;
using detail::id_;
using detail::id_ref;
using detail::data_ref;

#define SYCL_ID_SUBSCRIPT_OP(dimension)			\
template <>										\
id_ref id_<dimension>::operator[](size_t n) {	\
	return id_ref(n, &values[n]);				\
}

SYCL_ID_SUBSCRIPT_OP(1)
SYCL_ID_SUBSCRIPT_OP(2)
SYCL_ID_SUBSCRIPT_OP(3)

#undef SYCL_ID_SUBSCRIPT_OP

#define SYCL_ID_OPERATOR(op)						\
	template <>										\
	data_ref id_<1>::operator op(size_t n) const {	\
		return id_ref(0, nullptr) op n;				\
	}

SYCL_ID_OPERATOR(+)
SYCL_ID_OPERATOR(-)
SYCL_ID_OPERATOR(*)
SYCL_ID_OPERATOR(/)
SYCL_ID_OPERATOR(%)

data_ref operator*(size_t n, id_<1> i) {
	return n * id_ref(0, nullptr);
}

#undef SYCL_ID_OPERATOR
