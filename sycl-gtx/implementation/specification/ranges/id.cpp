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

#define SYCL_ID_OPERATOR(op)										\
	template <>														\
	detail::data_ref detail::id_<1>::operator op(size_t n) const {	\
		return id_ref(0, nullptr) op n;								\
	}

SYCL_ID_OPERATOR(+)

#undef SYCL_ID_OPERATOR
