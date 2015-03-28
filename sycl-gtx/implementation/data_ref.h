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
public:
	data_ref(string_class name)
		: name(name) {}
	const data_ref& operator=(int n) const;

	template <int dimensions>
	const data_ref& operator=(id<dimensions> index) const {
		DSELF() << "not implemented";
		kernel_::source::add(name + " = " + kernel_::source::get_name(index));
		return *this;
	}
};

} // namespace detail
} // namespace sycl
} // namespace cl
