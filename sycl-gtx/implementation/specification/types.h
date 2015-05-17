#pragma once

// 3.9.1 Description of the built-in types available for SYCL host and device

#include "../common.h"
#include "../counter.h"
#include "../data_ref.h"

namespace cl {
namespace sycl {

namespace detail {

template <typename dataT, int numElements>
class cl_type : public counter<cl_type<dataT, numElements>>, public data_ref {
private:
	static string_class type_name() {
		return type_string<dataT>() + (numElements == 1 ? "" : std::to_string(numElements));
	}

	string_class generate_name() const {
		return '_' + type_name() + '_' + std::to_string(counter_id);
	}

public:
	cl_type()
		: counter(), data_ref(generate_name()) {
		kernel_add(type_name() + ' ' + name);
	}

	template <class T>
	cl_type(T n)
		: counter(), data_ref(generate_name()) {
		kernel_add(type_name() + ' ' + name + " = " + get_name(n));
	}
};

} // namespace detail

#define SYCL_CL_TYPE_INHERIT(type)				\
	type()										\
		: Base() {}								\
	template <class T>							\
	type(T n)									\
		: Base(n) {}							\
	template <class T>							\
	data_ref& operator=(T n) {					\
		return data_ref::operator=(n);			\
	}											\
	template <class T>							\
	friend data_ref operator*(T n, type elem) {	\
		return n * (data_ref&&)elem;			\
	}

#define SYCL_CL_TYPE_SIGNED(type, numElements)								\
	class type##numElements : public detail::cl_type<type, numElements> {	\
		using Base = detail::cl_type<type, numElements>;					\
		using data_ref = detail::data_ref;									\
	public:																	\
		SYCL_CL_TYPE_INHERIT(type##numElements)								\
	};

#define SYCL_CL_TYPE_UNSIGNED(type, numElements)										\
	class u##type##numElements : public detail::cl_type<unsigned type, numElements> {	\
		using Base = detail::cl_type<unsigned type, numElements>;						\
		using data_ref = detail::data_ref;												\
	public:																				\
		SYCL_CL_TYPE_INHERIT(u##type##numElements)										\
	};

#define SYCL_ADD_SIGNED_CL_TYPE(type)	\
	SYCL_CL_TYPE_SIGNED(type, 1)		\
	SYCL_CL_TYPE_SIGNED(type, 2)		\
	SYCL_CL_TYPE_SIGNED(type, 3)		\
	SYCL_CL_TYPE_SIGNED(type, 4)		\
	SYCL_CL_TYPE_SIGNED(type, 8)		\
	SYCL_CL_TYPE_SIGNED(type, 16)

#define SYCL_ADD_UNSIGNED_CL_TYPE(type)	\
	SYCL_CL_TYPE_UNSIGNED(type, 1)		\
	SYCL_CL_TYPE_UNSIGNED(type, 2)		\
	SYCL_CL_TYPE_UNSIGNED(type, 3)		\
	SYCL_CL_TYPE_UNSIGNED(type, 4)		\
	SYCL_CL_TYPE_UNSIGNED(type, 8)		\
	SYCL_CL_TYPE_UNSIGNED(type, 16)

SYCL_ADD_SIGNED_CL_TYPE(bool)
SYCL_ADD_SIGNED_CL_TYPE(int)
SYCL_ADD_SIGNED_CL_TYPE(char)
SYCL_ADD_SIGNED_CL_TYPE(short)
SYCL_ADD_SIGNED_CL_TYPE(long)
SYCL_ADD_SIGNED_CL_TYPE(float)
SYCL_ADD_SIGNED_CL_TYPE(double)

SYCL_ADD_UNSIGNED_CL_TYPE(int)
SYCL_ADD_UNSIGNED_CL_TYPE(char)
SYCL_ADD_UNSIGNED_CL_TYPE(short)
SYCL_ADD_UNSIGNED_CL_TYPE(long)

#undef SYCL_ADD_SIGNED_CL_TYPE
#undef SYCL_ADD_UNSIGNED_CL_TYPE
#undef SYCL_CL_TYPE_INHERIT
#undef SYCL_CL_TYPE_SIGNED
#undef SYCL_CL_TYPE_UNSIGNED

} // namespace sycl
} // namespace cl
