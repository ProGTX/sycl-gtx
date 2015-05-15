#pragma once

// 3.6.4 Accessors

#include "access.h"
#include "../common.h"

namespace cl {
namespace sycl {

namespace detail {

// Forward declaration
namespace kernel_ {
	class source;
}

class accessor_base {
};

// 3.6.4.3 Core accessors class
template <typename DataType, int dimensions, access::mode mode, access::target target>
class accessor_core : public accessor_base {
public:
	// Returns the size of the underlying buffer in number of elements.
	size_t get_size() const;

	// Returns the cl_mem object corresponding to the access.
	// TODO: Only available when target is cl_image or cl_buffer
	virtual cl_mem get_cl_mem_object() const {
		return nullptr;
	}

	// Returns the cl_event object corresponding to the last command to access the memory object.
	// TODO: Only available when target is cl_image or cl_buffer.
	cl_event get_cl_event_object() const;

protected:
	friend class kernel_::source;

	virtual void* resource() const {
		return nullptr;
	}

	virtual size_t argument_size() const {
		return 0;
	}
};

template<bool>
struct select_target;

// This does not compile with enums (at least in MSVC 2013), use ints instead
template <typename DataType, int dimensions, int mode, int target, typename = select_target<true>>
class accessor_;

#define SYCL_ACCESSOR_CLASS(condition)																\
	template <typename DataType, int dimensions, int mode, int target>								\
	class accessor_<DataType, dimensions, mode, target, select_target<(condition)>>					\
		: public accessor_core<DataType, dimensions, (access::mode)mode, (access::target)target>

} // namespace detail


template <typename DataType, int dimensions, access::mode mode, access::target target = access::global_buffer>
class accessor;

#define SYCL_ADD_ACCESSOR(mode, target)									\
	template <typename DataType, int dimensions>						\
	class accessor<DataType, dimensions, mode, target>					\
		: public detail::accessor_<DataType, dimensions, mode, target>

} // namespace sycl
} // namespace cl
