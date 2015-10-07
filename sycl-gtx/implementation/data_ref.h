#pragma once

#include "common.h"
#include "debug.h"
#include <type_traits>

// Data reference wrappers

namespace cl {
namespace sycl {

// Forward declaration
template <int dimensions>
struct id;

namespace detail {

// Forward declaration
void kernel_add(string_class line);

class data_ref {
public:
	enum class type_t {
		general,
		numeric,
		id_global,
		id_local,
		range_global,
		range_local
	};

	static const string_class open_parenthesis;
	string_class name;
	type_t type;

	static string_class get_name(const data_ref& dref) {
		return dref.name;
	}

	template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
	static string_class get_name(T n) {
		return std::to_string(n);
	}

	data_ref(string_class name)
		: name(name) {}

	data_ref(const char* name)
		: name(name) {}

	template <class T>
	data_ref(T type)
		: name(get_name(type)) {}

	data_ref(const data_ref& copy) = default;
#if MSVC_LOW
	data_ref(data_ref&& move)
		: SYCL_MOVE_INIT(name) {}
	friend void swap(data_ref& first, data_ref& second) {
		using std::swap;
		SYCL_SWAP(name);
	}
#else
	data_ref(data_ref&&) = default;
#endif

	// Without this one explicitly stated, default copy assignment is used
	data_ref& operator=(data_ref dref) {
		kernel_add(name + " = " + dref.name);
		return *this;
	}


#define SYCL_ASSIGNMENT_OPERATOR(op)					\
	template <class T>									\
	data_ref& operator op(T n) {			 			\
		kernel_add(name + " " #op " " + get_name(n));	\
		return *this;						 			\
	}

	SYCL_ASSIGNMENT_OPERATOR(=);
	SYCL_ASSIGNMENT_OPERATOR(+=);
	SYCL_ASSIGNMENT_OPERATOR(-=);
	SYCL_ASSIGNMENT_OPERATOR(*=);
	SYCL_ASSIGNMENT_OPERATOR(/=);
	SYCL_ASSIGNMENT_OPERATOR(%=);

#undef SYCL_ASSIGNMENT_OPERATOR


	// TODO: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/operators.html

#define SYCL_DATA_REF_OPERATOR(op)																	\
	template <class T>																				\
	data_ref operator op(T n) const {																\
		return data_ref(open_parenthesis + name + " " #op " " + get_name(n) + ")");					\
	}																								\
	template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>	\
	friend data_ref operator op(T n, data_ref dref) {												\
		return data_ref(open_parenthesis + get_name(n) + " " #op " " + dref.name + ")");			\
	}

	// Arithmetic operatos
	SYCL_DATA_REF_OPERATOR(+);
	SYCL_DATA_REF_OPERATOR(-);
	SYCL_DATA_REF_OPERATOR(*);
	SYCL_DATA_REF_OPERATOR(/);
	SYCL_DATA_REF_OPERATOR(%);

	// Comparison operators
	SYCL_DATA_REF_OPERATOR(==);
	SYCL_DATA_REF_OPERATOR(!=);
	SYCL_DATA_REF_OPERATOR(<);
	SYCL_DATA_REF_OPERATOR(<=);
	SYCL_DATA_REF_OPERATOR(>);
	SYCL_DATA_REF_OPERATOR(>=);

	// Boolean operators
	SYCL_DATA_REF_OPERATOR(||);
	SYCL_DATA_REF_OPERATOR(&&);

#undef SYCL_DATA_REF_OPERATOR

};

} // namespace detail

} // namespace sycl
} // namespace cl
