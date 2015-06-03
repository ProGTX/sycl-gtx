#pragma once

#include "info.h"
#include "../common.h"
#include <CL/cl.h>

namespace cl {
namespace sycl {

namespace detail {

template <typename EnumClass, EnumClass Value, typename ReturnType, typename CLType>
struct param_traits2 {
	using type = ReturnType;
	using cl_type = CLType;
};

} // namespace detail

template <typename EnumClass, EnumClass Value>
struct param_traits2;

#define SYCL_ADD_TRAIT(EnumClass, Value, ReturnType, CLType)	\
template <>																	\
struct param_traits2<EnumClass, Value>										\
	: detail::param_traits2<												\
		EnumClass, Value, ReturnType, CLType					\
	> {};


// 3.3.3.2 Context information descriptors
// https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetContextInfo.html

#define SYCL_ADD_CONTEXT_TRAIT(Value, ReturnType)	\
	SYCL_ADD_TRAIT(info::context, Value, ReturnType, cl_context_info)

SYCL_ADD_CONTEXT_TRAIT(info::context::reference_count,	cl_uint)
SYCL_ADD_CONTEXT_TRAIT(info::context::num_devices,		cl_uint)
SYCL_ADD_CONTEXT_TRAIT(info::context::devices,			vector_class<cl_device_id>)
SYCL_ADD_CONTEXT_TRAIT(info::context::gl_interop,		info::gl_context_interop)

#undef SYCL_ADD_CONTEXT_TRAIT

#undef SYCL_ADD_TRAIT

} // namespace sycl
} // namespace cl
