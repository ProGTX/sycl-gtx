#include "id.h"
#include "../../data_ref.h"

using namespace cl::sycl;

template <int dimensions>
const string_class detail::id_<dimensions>::base_name = "_sycl_id";

#define FUNC_DECLARE(return_type, func_name, code)	\
	template <>										\
	return_type detail::id_<1>:: func_name code		\
	template <>										\
	return_type detail::id_<2>:: func_name code		\
	template <>										\
	return_type detail::id_<3>:: func_name code

FUNC_DECLARE(detail::id_ref, operator[](size_t n), {
	return id_ref(2, &values[n]);
})
