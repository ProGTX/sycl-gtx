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
	static const string_class open_parenthesis;
public:
	string_class name;

	data_ref(string_class name)
		: name(name) {}

	data_ref& operator=(int n);
	data_ref& operator=(id<1> index);
	data_ref& operator=(data_ref dref);

	data_ref operator+(data_ref dref) const;
	data_ref operator-(data_ref dref) const;

	data_ref operator-(int n) const;
	friend data_ref operator-(int n, data_ref dref) {
		return data_ref(open_parenthesis + "- " + dref.operator-(n).name + ")");
	}
	data_ref operator-(unsigned int n) const {
		return operator-((int)n);
	}
	friend data_ref operator-(unsigned int n, data_ref dref) {
		return data_ref(open_parenthesis + "- " + dref.operator-(n).name + ")");
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

} // namespace detail
} // namespace sycl
} // namespace cl
