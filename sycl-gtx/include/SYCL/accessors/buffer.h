#pragma once

// 3.4.6.4 Buffer accessors

#include "SYCL/accessor.h"
#include "SYCL/accessors/buffer_device.h"
#include "SYCL/accessors/buffer_host.h"
#include "SYCL/ranges.h"

namespace cl {
namespace sycl {

// Forward declarations
template <typename DataType, int dimensions>
struct buffer;
class handler;

#if MSVC_2013_OR_LOWER
#define SYCL_ADD_ACCESSOR_BUFFER(mode, target)                                \
  SYCL_ADD_ACCESSOR(mode, target) {                                           \
    using Base = detail::accessor_detail<DataType, dimensions, mode, target>; \
                                                                              \
   public:                                                                    \
    accessor(buffer<DataType, dimensions>& bufferRef,                         \
             handler& commandGroupHandler)                                    \
        : Base(bufferRef, commandGroupHandler) {}                             \
    accessor(buffer<DataType, dimensions>& bufferRef,                         \
             handler& commandGroupHandler, range<dimensions> offset,          \
             range<dimensions> range)                                         \
        : Base(bufferRef, commandGroupHandler, offset, range) {}              \
    accessor(Base&& move) : Base(std::move(move)) {}                          \
  };
#define SYCL_ADD_ACCESSOR_HOST_BUFFER(mode)                                \
  SYCL_ADD_ACCESSOR(mode, access::target::host_buffer) {                   \
    using Base = detail::accessor_detail<DataType, dimensions, mode,       \
                                         access::target::host_buffer>;     \
                                                                           \
   public:                                                                 \
    accessor(buffer<DataType, dimensions>& bufferRef) : Base(bufferRef) {} \
    accessor(buffer<DataType, dimensions>& bufferRef,                      \
             range<dimensions> offset, range<dimensions> range)            \
        : Base(bufferRef, offset, range) {}                                \
    accessor(Base&& move) : Base(std::move(move)) {}                       \
  };
#else
#define SYCL_ADD_ACCESSOR_BUFFER(mode, target)                                \
  SYCL_ADD_ACCESSOR(mode, target) {                                           \
    using Base = detail::accessor_detail<DataType, dimensions, mode, target>; \
                                                                              \
   public:                                                                    \
    using Base::Base;                                                         \
  };
#define SYCL_ADD_ACCESSOR_HOST_BUFFER(mode)                            \
  SYCL_ADD_ACCESSOR(mode, access::target::host_buffer) {               \
    using Base = detail::accessor_detail<DataType, dimensions, mode,   \
                                         access::target::host_buffer>; \
                                                                       \
   public:                                                             \
    using Base::Base;                                                  \
  };
#endif

/**
 * 3.4.6.8 Accessor capabilities and restrictions
 */
#define SYCL_ADD_ACC_BUFFERS(mode)                              \
  SYCL_ADD_ACCESSOR_BUFFER(mode, access::target::global_buffer) \
  SYCL_ADD_ACCESSOR_HOST_BUFFER(mode)

SYCL_ADD_ACC_BUFFERS(access::mode::read)
SYCL_ADD_ACC_BUFFERS(access::mode::write)
SYCL_ADD_ACC_BUFFERS(access::mode::read_write)
SYCL_ADD_ACC_BUFFERS(access::mode::discard_write)
SYCL_ADD_ACC_BUFFERS(access::mode::discard_read_write)

/** Can only be read */
SYCL_ADD_ACCESSOR_BUFFER(access::mode::read, access::target::constant_buffer)

}  // namespace sycl
}  // namespace cl

#undef SYCL_ADD_ACCESSOR_BUFFER
#undef SYCL_ADD_ACCESSOR_HOST_BUFFER
#undef SYCL_ADD_ACC_BUFFERS
