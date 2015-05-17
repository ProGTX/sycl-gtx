#pragma once

// 3.9.1 Description of the built-in types available for SYCL host and device

#include "../common.h"
#include "../counter.h"
#include "../data_ref.h"

namespace cl {
namespace sycl {

namespace detail {

template <typename dataT, int numElements, bool is_signed>
class cl_type : public counter<cl_type<dataT, numElements, is_signed>>, public data_ref {
private:
	static string_class type_name() {
		return string_class(is_signed ? "" : "u") + type_string<dataT>() + (numElements == 1 ? "" : std::to_string(numElements));
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

#define SYCL_CL_TYPE_SIGNED(type, numElements)									\
	class type##numElements : public detail::cl_type<type, numElements, true> {	\
		using Base = detail::cl_type<type, numElements, true>;					\
		using data_ref = detail::data_ref;										\
	public:																		\
		SYCL_CL_TYPE_INHERIT(type##numElements)									\
	};

#define SYCL_CL_TYPE_UNSIGNED(type, numElements)									\
	class u##type##numElements : public detail::cl_type<type, numElements, false> {	\
		using Base = detail::cl_type<type, numElements, false>;						\
		using data_ref = detail::data_ref;											\
	public:																			\
		SYCL_CL_TYPE_INHERIT(u##type##numElements)									\
	};

SYCL_CL_TYPE_SIGNED(int, 1)
SYCL_CL_TYPE_UNSIGNED(int, 1)
SYCL_CL_TYPE_SIGNED(int, 2)
SYCL_CL_TYPE_UNSIGNED(int, 2)

#undef SYCL_CL_TYPE_INHERIT
#undef SYCL_CL_TYPE_SIGNED
#undef SYCL_CL_TYPE_UNSIGNED

} // namespace sycl
} // namespace cl
