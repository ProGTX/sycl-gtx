#pragma once

#include "common.h"

namespace cl {
namespace sycl {

template<typename cl_struct, cl_int name>
struct param_traits;

template<cl_int name>
struct param_traits<cl_platform_info, name> {
	// https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clGetPlatformInfo.html lists all types as char[]
	// But one cannot safely return char[] from a function
	using param_type = STRING_CLASS;
};

} // namespace sycl
} // namespace cl
