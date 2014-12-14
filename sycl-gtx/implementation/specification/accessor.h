#pragma once

// 3.6.4 Accessors

#include "access.h"
#include "ranges.h"

namespace cl {
namespace sycl {

// Forward declarations
template <typename DataType, int dimensions>
struct buffer;

// Data reference wrappers
template <typename DataType>
class __atomic_ref;
template <typename DataType>
class __read_ref;

template <typename DataType>
class __write_ref {
public:
	__write_ref& operator=(int n) {
		return *this;
	}
};

namespace detail {

// 3.6.4.3 Core accessors class
template <typename DataType, int dimensions, access::mode mode, access::target target>
class accessor_core {
public:
	// Returns the size of the underlying buffer in number of elements.
	size_t get_size() const;

	// Returns the cl_mem object corresponding to the access.
	// TODO: Only available when target is cl_image or cl_buffer
	cl_mem get_cl_mem_object() const;

	// Returns the cl_event object corresponding to the last command to access the memory object.
	// TODO: Only available when target is cl_image or cl_buffer.
	cl_event get_cl_event_object() const;
};

template<bool>
struct select_target;

// This does not compile with enums (at least in MSVC 2013), use ints instead
template <typename DataType, int dimensions, int mode, int target, typename = select_target<true>>
class accessor_;

#define SYCL_ACCESSOR_CLASS(condition)															\
template <typename DataType, int dimensions, int mode, int target>								\
class accessor_<DataType, dimensions, mode, target, select_target<(condition)>>					\
	: public accessor_core<DataType, dimensions, (access::mode)mode, (access::target)target>

// 3.6.4.4 Buffer accessors
SYCL_ACCESSOR_CLASS(target == access::global_buffer || target == access::constant_buffer || target == access::host_buffer) {
public:
	accessor_(cl::sycl::buffer<DataType, dimensions>& targette) {
		DSELF() << "not implemented";
	}
};

} // namespace detail


template <typename DataType, int dimensions, access::mode mode, access::target target = access::global_buffer>
class accessor;

#define SYCL_ADD_ACCESSOR(mode)												\
	template <typename DataType, int dimensions, access::target target>		\
	class accessor<DataType, dimensions, mode, target>						\
		: public detail::accessor_<DataType, dimensions, mode, target>


// 3.6.4.4 Buffer accessors

SYCL_ADD_ACCESSOR(access::read) {
public:
#if MSVC_LOW
	accessor(buffer<DataType, dimensions>& targette)
		: detail::accessor_<DataType, dimensions, access::read, target>(targette) {}
#else
	using detail::accessor_<DataType, dimensions, access::read, target>::accessor_;
#endif
	// Read element from target data.
	__read_ref<DataType> operator[](id<dimensions>) const {
		DSELF() << "not implemented";
		return __read_ref<DataType>();
	}
};

SYCL_ADD_ACCESSOR(access::write) {
public:
	accessor(buffer<DataType, dimensions>& targette)
		: detail::accessor_<DataType, dimensions, access::write, target>(targette) {}
	// Reference to target element.
	__write_ref<DataType> operator[](id<dimensions> index) const {
		DSELF() << "not implemented";
		return __write_ref<DataType>();
	}
};

SYCL_ADD_ACCESSOR(access::atomic) {
public:
	accessor(buffer<DataType, dimensions>& targette)
		: detail::accessor_<DataType, dimensions, access::atomic, target>(targette) {}
	// Atomic reference to element from target data.
	__atomic_ref<DataType> operator[](id<dimensions>) const;
};

} // namespace sycl
} // namespace cl

#undef SYCL_ACCESSOR_CLASS
#undef SYCL_ADD_ACCESSOR
