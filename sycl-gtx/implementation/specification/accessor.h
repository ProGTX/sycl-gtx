#pragma once

// 3.4.6 Accessors

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


#if MSVC_LOW
// Indirection required because MSVC2013 fails on enum parameter SFINAE
// The final accessor class also needs a move constructor from base type: accessor(accessor_&&)
using acc_mode_t = int;
using acc_target_t = int;
#else
using acc_mode_t = access::mode;
using acc_target_t = access::target;
#endif


// 3.6.4.3 Core accessors class
template <typename DataType, int dimensions, access::mode mode, access::target target>
class accessor_core : public accessor_base {
public:
	using value_type = DataType;
	using reference = value_type&;
	using const_reference = const value_type&;

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

template <bool>
struct select_target;

// This does not compile with enums (at least in MSVC 2013), use ints instead
template <typename DataType, int dimensions, acc_mode_t mode, acc_target_t target, typename = select_target<true>>
class accessor_;

#define SYCL_ACCESSOR_CLASS(condition)																\
	template <typename DataType, int dimensions, acc_mode_t mode, acc_target_t target>				\
	class accessor_<DataType, dimensions, mode, target, select_target<(condition)>>					\
		: public accessor_core<DataType, dimensions, (access::mode)mode, (access::target)target>

} // namespace detail


template <typename DataType, int dimensions = 1, access::mode mode = access::read_write, access::target target = access::global_buffer>
class accessor;

#define SYCL_ADD_ACCESSOR(mode, target)								\
	template <typename DataType, int dimensions>					\
	class accessor<DataType, dimensions, mode, target>				\
		: public detail::accessor_<									\
			DataType, dimensions,									\
			(detail::acc_mode_t)mode, (detail::acc_target_t)target>

} // namespace sycl
} // namespace cl
