#pragma once

#include "specification\ranges.h"

#include "common.h"
#include "debug.h"


namespace cl {
namespace sycl {

namespace detail {

// Data reference wrappers
class __atomic_ref;
class __read_ref;
class __write_ref {
private:
	string_class name;
public:
	__write_ref(string_class name)
		: name(name) {}
	const __write_ref& operator=(int n) const;

	template <int dimensions>
	const __write_ref& operator=(id<dimensions> index) const {
		DSELF() << "not implemented";
		kernel_::source::add(name + " = " + kernel_::source::get_name(index));
		return *this;
	}
};

} // namespace detail

} // namespace sycl
} // namespace cl
