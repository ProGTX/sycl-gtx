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

#define SYCL_DATA_REF_OPERATOR(op)																	\
	template <class T, is_compatible_t<T>* = nullptr>												\
	data_ref operator op(T n) const {																\
		return data_ref(open_parenthesis + name + " " #op " " + get_name(n) + ")");					\
	}																								\
	template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>	\
	friend data_ref operator op(T n, data_ref dref) {												\
		return data_ref(open_parenthesis + get_name(n) + " " #op " " + dref.name + ")");			\
	}

#define SYCL_ASSIGNMENT_OPERATOR(op, mode)				\
	template <class T, is_compatible_t<T>* = nullptr>	\
	data_ref& operator op(T n) {						\
		return assign_<assign::mode>(n);				\
	}

class data_ref {
public:
	template <class T>
	using is_compatible_t = typename std::enable_if<
		std::is_arithmetic<T>::value									||
		std::is_same<id<1>, typename std::decay<T>::type>::value		||
		std::is_base_of<data_ref, typename std::decay<T>::type>::value
	>::type;

	static const string_class open_parenthesis;
	string_class name;

	data_ref(string_class name)
		: name(name) {}

private:
	struct assign {
		static const char normal[];
		static const char add[];
		static const char subtract[];
		static const char multiply[];
		static const char divide[];
		static const char modulo[];
	};

	void kernel_add(string_class line);

	template <const char* op, class T, is_compatible_t<T>* = nullptr>
	data_ref& assign_(T n) {
		kernel_add(name + op + get_name(n));
		return *this;
	}

public:
	// Without this one explicitly stated, default copy assignment is used
	data_ref& operator=(data_ref dref) {
		return assign_<assign::normal>(dref);
	}

	SYCL_ASSIGNMENT_OPERATOR(= , normal);
	SYCL_ASSIGNMENT_OPERATOR(+= , add);
	SYCL_ASSIGNMENT_OPERATOR(-= , subtract);
	SYCL_ASSIGNMENT_OPERATOR(*= , multiply);
	SYCL_ASSIGNMENT_OPERATOR(/= , divide);
	SYCL_ASSIGNMENT_OPERATOR(%= , modulo);

	static string_class get_name(id<1> index);
	static string_class get_name(id<2> index);
	static string_class get_name(id<3> index);

	static string_class get_name(data_ref dref) {
		return dref.name;
	}
	template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
	static string_class get_name(T n) {
		return std::to_string(n);
	}

	// TODO: https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/operators.html

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
};

class id_ref : public data_ref {
protected:
	size_t* value;
public:
	id_ref(int n, size_t* value);

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

#undef SYCL_DATA_REF_OPERATOR
#undef SYCL_ASSIGNMENT_OPERATOR
