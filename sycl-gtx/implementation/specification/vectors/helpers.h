#pragma once

#include "../../common.h"


namespace cl {
namespace sycl {

// Forward declaration
template <typename dataT, int numElements>
class vec;


namespace detail {

namespace vectors {
// Forward declaration
template <typename dataT, int numElements>
struct data;
}


// These are defined elsewhere, here only specialization for vectors

template <typename dataT, int numElements>
struct data_size<vec<dataT, numElements>> {
	static size_t get() {
		return sizeof(typename base_host_data<vec<dataT, numElements>>::type);
	}
};

template <typename dataT, int numElements>
struct base_host_data<vec<dataT, numElements>> {
	using type = typename vectors::data<dataT, numElements>::type;
};

template <typename dataT, int numElements>
struct acc_device_return<vec<dataT, numElements>> {
	using type = vec<dataT, numElements>;
};

// Cannot be joined with vector declarations
// because a vector of 3 is a typedef of a vector of 4
#define SYCL_CY_TYPE_STRING(nummedType)	\
template <>								\
struct type_string<cl_##nummedType> {	\
	static string_class get() {			\
		return #nummedType;				\
	}									\
};

#define SYCL_ADD_CL_TYPE_STRING(basetype)	\
	SYCL_CY_TYPE_STRING(basetype##2)		\
	SYCL_CY_TYPE_STRING(basetype##4)		\
	SYCL_CY_TYPE_STRING(basetype##8)		\
	SYCL_CY_TYPE_STRING(basetype##16)

SYCL_ADD_CL_TYPE_STRING(int)
SYCL_ADD_CL_TYPE_STRING(char)
SYCL_ADD_CL_TYPE_STRING(short)
SYCL_ADD_CL_TYPE_STRING(long)
SYCL_ADD_CL_TYPE_STRING(float)
SYCL_ADD_CL_TYPE_STRING(double)

SYCL_ADD_CL_TYPE_STRING(uint)
SYCL_ADD_CL_TYPE_STRING(uchar)
SYCL_ADD_CL_TYPE_STRING(ushort)
SYCL_ADD_CL_TYPE_STRING(ulong)

#undef SYCL_ADD_CL_TYPE_STRING
#undef SYCL_CY_TYPE_STRING

} // namespace detail

} // namespace sycl
} // namespace cl
