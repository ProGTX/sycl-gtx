#pragma once

#include "common.h"
#include "debug.h"

// Data reference wrappers

namespace cl {
namespace sycl {

// Forward declaration
template <int dimensions>
class id;

namespace detail {

class data_ref {
protected:
	bool assignable = true;

public:
	string_class name;

	data_ref(string_class name)
		: name(name) {}

	data_ref& operator=(int n);
	data_ref& operator=(id<1> index);
	data_ref& operator=(data_ref dref);
	data_ref& operator+(data_ref dref);
	data_ref& operator*(int n);
};

class id_ref : public data_ref {
protected:
	size_t* value;
public:
	id_ref(int dimensions, size_t* value)
		: data_ref(id_base_name + std::to_string(dimensions - 1)), value(value) {}

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
