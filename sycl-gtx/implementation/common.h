#pragma once

#include <CL/cl.h>

#if _MSC_VER <= 1800
#define MSVC_LOW 1
#endif

#if MSVC_LOW
#define SYCL_MOVE_OPS(class_name, action)									\
private:																	\
void swap(class_name& first, class_name& second) action						\
public:																		\
	class_name(class_name&& move) { swap(*this, move); }					\
	class_name operator=(class_name&& move) { swap(*this, move); return *this; }
#define SYCL_COPY(member) first.member = second.member
#define SYCL_MOVE(member) first.member = std::move(second.member)
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

// Meant for transforming arrays of pointers to OpenCL structs into a vector
template<class Container, class Inner>
VECTOR_CLASS<Container> to_vector(Inner* array, cl_uint new_size, bool) {
	VECTOR_CLASS<Container> vector;
	vector.reserve(new_size);
	for(cl_uint i = 0; i < new_size; ++i) {
		vector.emplace_back(array[i]);
	}
	return vector;
}
template<class Container, class Inner, size_t ArraySize>
VECTOR_CLASS<Container> to_vector(Inner(&array)[ArraySize], cl_uint size) {
	return to_vector<Container>(array, size, true);
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
