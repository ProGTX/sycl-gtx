#pragma once

#include "common.h"

namespace cl {
namespace sycl {
namespace detail {

template <class T, unsigned int start = 0>
class counter {
private:
	static unsigned int internal_count;
protected:
	const unsigned int counter_id;

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

template <class T, unsigned int start>
unsigned int counter<T, start>::internal_count = start;

} // namespace detail
} // namespace sycl
} // namespace cl
