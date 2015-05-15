#pragma once

#include "common.h"

namespace cl {
namespace sycl {
namespace detail {

template <int start = 0>
class counter {
private:
	static int internal_count;
protected:
	const int counter_id;

	counter()
		: counter_id(internal_count++) {}

	counter(const counter& copy) = default;
#if MSVC_LOW
	counter(counter&& move)
		: counter_id(counter_id) {}
#else
	counter(counter&&) = default;
#endif
};

template <int start>
int counter<start>::internal_count = start;

} // namespace detail
} // namespace sycl
} // namespace cl
