#pragma once

#include <memory>
#include <CL/cl.h>

#if _MSC_VER <= 1800
#define MSVC_LOW 1
#endif

#if MSVC_LOW
#define SYCL_SWAP(member) swap(first.member, second.member)
#define SYCL_MOVE_INIT(member) member(std::move(move.member))
#define SYCL_THREAD_LOCAL __declspec(thread)
#else
#define SYCL_THREAD_LOCAL thread_local
#endif

// 3.3 Vector, string and function classes in interfaces

#ifndef CL_SYCL_NO_STD_VECTOR
#include <vector>
#endif

#ifndef CL_SYCL_NO_STD_STRING
#include <string>
#endif

#ifndef CL_SYCL_NO_STD_FUNCTION
#include <functional>
#endif

namespace cl {
namespace sycl {

#ifndef CL_SYCL_NO_STD_VECTOR
	template<class T, class Alloc = ::std::allocator<T>>
	using vector_class = ::std::vector<T, Alloc>;
#endif

#ifndef CL_SYCL_NO_STD_STRING
	using string_class = ::std::string;
#endif

#ifndef CL_SYCL_NO_STD_FUNCTION
#if MSVC_LOW
	template<class... Args>
	class function_class : public ::std::function<void(Args...)> {
	private:
		using Base = ::std::function<void(Args...)>;
	public:
		function_class() {}
		function_class(nullptr_t fn)
			: Base(fn) {}
		template<class Fn>
		function_class(Fn fn)
			: Base(fn) {}
		function_class(const Base& x)
			: Base(x) {}
		function_class(Base&& x)
			: Base(std::move(x)) {}
		function_class(const function_class&) = default;
		function_class(function_class&& move)
			: Base((Base&&)std::move(move)) {}
	};
#else
	template<class... Args>
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

template <class T>
using shared_unique = std::shared_ptr<std::unique_ptr<T>>;

template <typename DataType>
static string_class type_string();

#define SYCL_GET_TYPE_STRING(type)			\
template <>									\
static string_class type_string<type>() {	\
	return #type "*";						\
}

SYCL_GET_TYPE_STRING(int)
SYCL_GET_TYPE_STRING(char)
SYCL_GET_TYPE_STRING(short)
SYCL_GET_TYPE_STRING(long)
SYCL_GET_TYPE_STRING(unsigned int)
SYCL_GET_TYPE_STRING(unsigned char)
SYCL_GET_TYPE_STRING(unsigned short)
SYCL_GET_TYPE_STRING(unsigned long)
SYCL_GET_TYPE_STRING(float)
SYCL_GET_TYPE_STRING(double)

#undef SYCL_GET_TYPE_STRING

} // namespace detail

} // namespace sycl
} // namespace cl
