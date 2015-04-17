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

// Forward declaration
class data_ref;

// http://stackoverflow.com/a/15598994/793006
template <typename T, bool = std::is_arithmetic<T>::value>
struct data_ref_name;
template <typename T>
struct data_ref_name<T, true> {
	static string_class get(T n);
};
template <typename T>
struct data_ref_name<T, false> {
	static string_class get(T dref);
};

class data_ref {
public:
	static const string_class open_parenthesis;
	string_class name;

	data_ref(string_class name)
		: name(name) {}

	data_ref& operator=(int n);
	data_ref& operator=(id<1> index);
	data_ref& operator=(data_ref dref);

	data_ref operator+(data_ref dref) const;

	template <typename T>
	data_ref operator-(T n) {
		return data_ref(open_parenthesis + name + " - " + data_ref_name<T>::get(n) + ")");
	}
	template <typename T>
	friend data_ref operator-(T n, data_ref dref) {
		return data_ref(open_parenthesis + data_ref_name<T>::get(n) + " - " + dref.name + ")");
	}

	data_ref operator*(int n) const;
	friend data_ref operator*(int n, data_ref dref) {
		return dref.operator*(n);
	}
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

template <typename T>
string_class data_ref_name<T, true>::get(T n) {
	return std::to_string(n);
}
template <typename T>
string_class data_ref_name<T, false>::get(T dref) {
	return dref.name;
}

} // namespace detail

} // namespace sycl
} // namespace cl
