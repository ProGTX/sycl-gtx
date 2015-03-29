#pragma once

#include "specification\ranges.h"

#include "common.h"
#include "debug.h"

// Data reference wrappers

namespace cl {
namespace sycl {
namespace detail {

class data_ref {
private:
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

} // namespace detail
} // namespace sycl
} // namespace cl
