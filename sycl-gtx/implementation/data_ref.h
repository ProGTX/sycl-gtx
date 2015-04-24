#pragma once

#include "common.h"
#include "debug.h"

#include <type_traits>

// Data reference wrappers

namespace cl {
namespace sycl {

// Forward declaration
template <int dimensions>
class id;

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
	};

	void kernel_add(string_class line);

	template <const char* op, class T, is_compatible_t<T>* = nullptr>
	data_ref& assign_(T n) {
		kernel_add(name + op + get_name(n));
		return *this;
	}

public:
	data_ref& operator=(data_ref dref) {
		return assign_<assign::normal>(dref);
	}
	template <class T, is_compatible_t<T>* = nullptr>
	data_ref& operator=(T n) {
		return assign_<assign::normal>(n);
	}

	static string_class get_name(id<1> index);

	static string_class get_name(data_ref dref) {
		return dref.name;
	}
	template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
	static string_class get_name(T n) {
		return std::to_string(n);
	}

	SYCL_DATA_REF_OPERATOR(-);
	SYCL_DATA_REF_OPERATOR(+);
	SYCL_DATA_REF_OPERATOR(*);
	SYCL_DATA_REF_OPERATOR(/);
};

class id_ref : public data_ref {
protected:
	size_t* value;
public:
	id_ref(int n, size_t* value)
		: data_ref(id_base_name + std::to_string(n)), value(value) {}

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
