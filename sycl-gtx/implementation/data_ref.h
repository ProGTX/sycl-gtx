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

// Forward declarations
void kernel_add(string_class line);
template <size_t dimensions, bool is_numeric>
struct point;

class data_ref {
public:
	enum class type_t {
		general,
		id_global,
		id_local,
		range_global,
		range_local
	};

	static const string_class open_parenthesis;
	string_class name;
	type_t type;

	static string_class get_name(id<1> index);
	static string_class get_name(id<2> index);
	static string_class get_name(id<3> index);

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

class point_ref : public data_ref {
protected:
	size_t* values;
public:
	template <size_t dimensions, bool is_numeric>
	point_ref(point<dimensions, is_numeric>* p)
		: data_ref(""), values(p->values) {
		type = p->type;

		switch(type) {
			case type_t::id_global:
				name = point_names::id_global;
				break;
			case type_t::id_local:
				name = point_names::id_local;
				break;
			case type_t::range_global:
				name = point_names::range_global;
				break;
			case type_t::range_local:
				name = point_names::range_local;
				break;
		}
	}

	// Use with caution
	template <bool defer = true>
	point_ref(point<1, true> p)
		: point_ref(&p) {}

	operator size_t&() {
		return *values;
	}

	operator size_t() const {
		return *values;
	}
};

class id_ref : public data_ref {
protected:
	size_t* value;
public:
	id_ref(int n, size_t* value, type_t access_type = type_t::id_global);

	operator size_t&() {
		return *value;
	}

	operator size_t() const {
		return *value;
	}
};

} // namespace detail

} // namespace sycl
} // namespace cl
