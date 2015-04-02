#include "id.h"
#include "../../data_ref.h"

using namespace cl::sycl;

template <int dimensions>
const string_class detail::id_<dimensions>::base_name = "_sycl_id";

#define SUBSCRIPT_OPERATOR(dimension)							\
template <>														\
detail::id_ref detail::id_<dimension>::operator[](size_t n) {	\
	return id_ref(dimension, &values[n]);						\
}

SUBSCRIPT_OPERATOR(1)
SUBSCRIPT_OPERATOR(2)
SUBSCRIPT_OPERATOR(3)
