#include "id.h"
#include "../../data_ref.h"

using namespace cl::sycl;

#define SYCL_ID_SUBSCRIPT_OP(dimension)							\
template <>														\
detail::id_ref detail::id_<dimension>::operator[](size_t n) {	\
	return id_ref(n, &values[n]);								\
}

SYCL_ID_SUBSCRIPT_OP(1)
SYCL_ID_SUBSCRIPT_OP(2)
SYCL_ID_SUBSCRIPT_OP(3)

#undef SYCL_ID_SUBSCRIPT_OP


template <>
detail::data_ref detail::id_<1>::operator+(size_t n) const {
	return id_ref(0, nullptr) + n;
}
