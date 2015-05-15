#pragma once

// 3.6.4.4 Buffer accessors

#include "buffer_device.h"
#include "buffer_host.h"
#include "../accessor.h"
#include "../ranges.h"

namespace cl {
namespace sycl {

// Forward declaration
template <typename DataType, int dimensions>
struct buffer;

#if MSVC_LOW
#define SYCL_ADD_ACCESSOR_BUFFER(mode, target)								\
	SYCL_ADD_ACCESSOR(mode, target) {										\
		using Base = detail::accessor_<DataType, dimensions, mode, target>;	\
	public:																	\
		accessor(buffer<DataType, dimensions>& bufferRef)					\
			: Base(bufferRef) {}											\
		accessor(															\
			buffer<DataType, dimensions>& bufferRef,						\
			range<dimensions> offset,										\
			range<dimensions> range											\
		)																	\
			: Base(bufferRef, offset, range) {}								\
	};
#else
#define SYCL_ADD_ACCESSOR_BUFFER(mode)										\
	SYCL_ADD_ACCESSOR(mode) {												\
		using Base = detail::accessor_<DataType, dimensions, mode, target>;	\
	public:																	\
		using Base::accessor_;												\
	};
#endif

// 3.6.4.9 Accessor capabilities and restrictions

#define SYCL_ADD_ACC_BUFFERS(mode)							\
	SYCL_ADD_ACCESSOR_BUFFER(mode, access::cl_buffer)		\
	SYCL_ADD_ACCESSOR_BUFFER(mode, access::global_buffer)	\
	SYCL_ADD_ACCESSOR_BUFFER(mode, access::host_buffer)

SYCL_ADD_ACC_BUFFERS(access::read)
SYCL_ADD_ACC_BUFFERS(access::write)
SYCL_ADD_ACC_BUFFERS(access::read_write)
SYCL_ADD_ACC_BUFFERS(access::discard_write)
SYCL_ADD_ACC_BUFFERS(access::discard_read_write)

// Can only be read
SYCL_ADD_ACCESSOR_BUFFER(access::read, access::constant_buffer)

} // namespace sycl
} // namespace cl

#undef SYCL_ADD_ACCESSOR_BUFFER
#undef SYCL_ADD_ACC_BUFFERS
