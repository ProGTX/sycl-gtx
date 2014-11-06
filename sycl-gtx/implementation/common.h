#pragma once

#include <CL/cl.h>

#if _MSC_VER <= 1800
#define MSVC_LOW 1
#endif

#if MSVC_LOW
#define SYCL_SWAP(member) swap(first.member, second.member)
#define SYCL_MOVE_INIT(member) member(std::move(move.member))
#endif

// 3.2.0 VECTOR_CLASS and STRING_CLASS

#if __NO_STD_VECTOR
#define VECTOR_CLASS cl::sycl::vector
#elif __USE_DEV_VECTOR
#define VECTOR_CLASS std::string
#else
#include <vector>
#define VECTOR_CLASS std::vector
#endif

#if __NO_STD_STRING
#define STRING_CLASS cl::sycl::string
#elif __USE_DEV_STRING
#define STRING_CLASS std::string
#else
#include <string>
#define STRING_CLASS std::string
#endif


namespace cl {
namespace sycl {
namespace helper {

template<class To, class From>
VECTOR_CLASS<To> transform_vector(VECTOR_CLASS<From> array) {
	return VECTOR_CLASS<To>(array.data(), array.data() + array.size());
}

template<cl_uint extension_macro, class T>
bool has_extension(T* sycl_class, const STRING_CLASS extension_name) {
	// TODO: Maybe add caching
	auto extensions = sycl_class->get_info<extension_macro>();
	STRING_CLASS ext_str(extensions);
	return ext_str.find(extension_name) != STRING_CLASS::npos;
}

} // namespace helper
} // namespace sycl
} // namespace cl
