#pragma once

// 3.3.4 Accessors

#include "access.h"
#include "buffer.h"
#include "ranges.h"

namespace cl {
namespace sycl {

template <typename dataType>
class __atomic_ref {};

// 3.3.4.4 Buffer accessors
template <typename dataType, int dimensions, access::mode mode, access::target target>
class accessor {
public:
	accessor(buffer<dataType, dimensions>& target) {}

	// Reference to target element.
	// Only if mode contains write access
	dataType& operator[](id<dimensions>) {}

	/*
	// Read element from target data.
	// Only if mode is read-only
	const dataType& operator[](id<dimensions>) {}

	// Atomic reference to element from target data.
	// Only if mode is atomic.
	__atomic_ref<dataType> operator[](id<dimensions>) {}
	*/
};

} // namespace sycl
} // namespace cl
