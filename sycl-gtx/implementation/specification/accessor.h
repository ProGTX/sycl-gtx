#pragma once

// 3.3.4 Accessors

#include "access.h"
#include "ranges.h"

namespace cl {
namespace sycl {

// Forward declaration
template <typename DataType, int dimensions>
struct buffer;

namespace detail {

template<bool>
struct select_target;

// 3.3.4.3 Core accessors class
template <typename dataType, int dimensions, access::mode mode, access::target target, typename = select_target<true>>
class accessor_core {
public:
	int get_size();
	cl_mem get_cl_mem_object();
	cl_event get_cl_event_object();
};


template <typename dataType, int dimensions, access::mode mode, access::target target>
class accessor_;

#define SYCL_ACCESSOR_CLASS(condition)																	\
	template <typename dataType, int dimensions, access::mode mode, access::target target>								\
	class accessor_ : public detail::accessor_core<dataType, dimensions, mode, target, detail::select_target<(condition)>>

// 3.3.4.4 Buffer accessors
SYCL_ACCESSOR_CLASS(target == access::global_buffer || target == access::constant_buffer || target == access::host_buffer) {
public:
	accessor_(buffer<dataType, dimensions>& targette) {
		DSELF() << "not implemented";
	}
	// Atomic reference to element from target data.
	// Only if mode is atomic.
	//__atomic_ref<dataType> operator[](id<dimensions>) {}
};

} // namespace detail


template <typename dataType, int dimensions, access::mode mode, access::target target = access::global_buffer>
class accessor;

#define SYCL_ACCESSOR_BUFFER(mode)																					\
	template <typename dataType, int dimensions, access::target target>												\
	class accessor<dataType, dimensions, mode, target> : public detail::accessor_<dataType, dimensions, mode, target>

SYCL_ACCESSOR_BUFFER(access::read) {
public:
	accessor(buffer<dataType, dimensions>& targette)
		: detail::accessor_<dataType, dimensions, access::read, target>(targettte) {}
	// Read element from target data.
	const dataType& operator[](id<dimensions>) {
		DSELF() << "not implemented";
	}
};

SYCL_ACCESSOR_BUFFER(access::write) {
public:
	accessor(buffer<dataType, dimensions>& targette)
		: detail::accessor_<dataType, dimensions, access::write, target>(targettte) {}
	// Reference to target element.
	dataType& operator[](id<dimensions>) {
		DSELF() << "not implemented";
	}
};

} // namespace sycl
} // namespace cl

#undef SYCL_ACCESSOR_CLASS
#undef SYCL_ACCESSOR_BUFFER
