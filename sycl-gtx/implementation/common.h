#pragma once

#include <CL/cl.h>

#if _MSC_VER <= 1800
#define MSVC_LOW 1
#endif

#if MSVC_LOW
#define SYCL_SWAP(member) swap(first.member, second.member)
#define SYCL_MOVE_INIT(member) member(std::move(move.member))
#endif

// 3.3 Vector, string and function classes in interfaces

#ifndef CL_SYCL_NO_STD_VECTOR
#include <vector>
#endif

#ifndef CL_SYCL_NO_STD_STRING
#include <vector>
#endif

#ifndef CL_SYCL_NO_STD_FUNCTION
#include <functional>
#endif

namespace cl {
namespace sycl {

#ifndef CL_SYCL_NO_STD_VECTOR
	template < class T, class Alloc = ::std::allocator<T> >
	using vector_class = ::std::vector<T, Alloc>;
#endif

#ifndef CL_SYCL_NO_STD_STRING
	using string_class = ::std::string;
#endif

#ifndef CL_SYCL_NO_STD_FUNCTION
	template<class... Args>
#if MSVC_LOW
	class function_class : public ::std::function<void(Args...)> {};
#else
	using function_class = ::std::function<void(Args...)>;
#endif
#endif


namespace detail {

template<class To, class From>
vector_class<To> transform_vector(vector_class<From> array) {
	return vector_class<To>(array.data(), array.data() + array.size());
}

template<cl_uint extension_macro, class T>
bool has_extension(T* sycl_class, const string_class extension_name) {
	// TODO: Maybe add caching
	auto extensions = sycl_class->get_info<extension_macro>();
	string_class ext_str(extensions);
	return ext_str.find(extension_name) != string_class::npos;
}

} // namespace detail
} // namespace sycl
} // namespace cl
