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
#define SYCL_ADD_ACCESSOR_BUFFER(mode)										\
	SYCL_ADD_ACCESSOR(mode) {												\
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

SYCL_ADD_ACCESSOR_BUFFER(access::read)
SYCL_ADD_ACCESSOR_BUFFER(access::write)
SYCL_ADD_ACCESSOR_BUFFER(access::read_write)
SYCL_ADD_ACCESSOR_BUFFER(access::discard_read_write)

} // namespace sycl
} // namespace cl

#undef SYCL_ACCESSOR_CLASS
#undef SYCL_ADD_ACCESSOR
#undef SYCL_ADD_ACCESSOR_BUFFER
#undef SYCL_BUFFER_CONSTRUCTORS
