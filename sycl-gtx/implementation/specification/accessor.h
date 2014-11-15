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
template <typename DataType, int dimensions, access::mode mode, access::target target, typename = select_target<true>>
class accessor_core {
public:
	int get_size();
	cl_mem get_cl_mem_object();
	cl_event get_cl_event_object();
};


template <typename DataType, int dimensions, access::mode mode, access::target target>
class accessor_;

#define SYCL_ACCESSOR_CLASS(condition)																	\
	template <typename DataType, int dimensions, access::mode mode, access::target target>								\
	class accessor_ : public detail::accessor_core<DataType, dimensions, mode, target, detail::select_target<(condition)>>

// 3.3.4.4 Buffer accessors
SYCL_ACCESSOR_CLASS(target == access::global_buffer || target == access::constant_buffer || target == access::host_buffer) {
public:
	accessor_(cl::sycl::buffer<DataType, dimensions>& targette) {
		DSELF() << "not implemented";
	}
	// Atomic reference to element from target data.
	// Only if mode is atomic.
	//__atomic_ref<DataType> operator[](id<dimensions>) {}
};

} // namespace detail


template <typename DataType, int dimensions, access::mode mode, access::target target = access::global_buffer>
class accessor;

#define SYCL_ACCESSOR_BUFFER(mode)																					\
	template <typename DataType, int dimensions, access::target target>												\
	class accessor<DataType, dimensions, mode, target> : public detail::accessor_<DataType, dimensions, mode, target>

SYCL_ACCESSOR_BUFFER(access::read) {
public:
	accessor(buffer<DataType, dimensions>& targette)
		: detail::accessor_<DataType, dimensions, access::read, target>(targette) {}
	// Read element from target data.
	const DataType& operator[](id<dimensions>) {
		DSELF() << "not implemented";
	}
};

SYCL_ACCESSOR_BUFFER(access::write) {
public:
	accessor(buffer<DataType, dimensions>& targette)
		: detail::accessor_<DataType, dimensions, access::write, target>(targette) {}
	// Reference to target element.
	DataType& operator[](id<dimensions>) {
		DSELF() << "not implemented";
	}
};

} // namespace sycl
} // namespace cl

#undef SYCL_ACCESSOR_CLASS
#undef SYCL_ACCESSOR_BUFFER
