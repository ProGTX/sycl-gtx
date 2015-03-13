#pragma once

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
};

} // namespace detail

} // namespace sycl
} // namespace cl
