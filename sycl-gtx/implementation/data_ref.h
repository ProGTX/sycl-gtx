#pragma once

#include "specification\ranges.h"

#include "common.h"
#include "debug.h"

// Data reference wrappers

namespace cl {
namespace sycl {
namespace detail {

class data_ref {
protected:
	string_class name;
	bool assignable = true;

public:
	data_ref(string_class name)
		: name(name) {}

	data_ref& operator=(int n);
	data_ref& operator=(data_ref dref);
	data_ref& operator+(data_ref dref);

	template <int dimensions>
	data_ref& operator=(id<dimensions> index) {
		DSELF() << "not implemented";
		if(assignable) {
			kernel_::source::add(name + " = " + kernel_::source::get_name(index));
		}
		else {
			// TODO: Error
		}
		return *this;
	}
};

class id_ref : public data_ref {
protected:
	size_t* value;
public:
	id_ref(int dimensions, size_t* value)
		: data_ref(string_class("_sycl_id") + std::to_string(dimensions)), value(value) {}

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
