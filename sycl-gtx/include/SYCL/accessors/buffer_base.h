#pragma once

#include "SYCL/detail/common.h"
#include "SYCL/ranges.h"

namespace cl {
namespace sycl {

// Forward declarations
template <typename DataType, int dimensions>
struct buffer;
class handler;

namespace detail {

/**
 * Core buffer accessor class
 *
 * 3.4.6 Accessors and 3.4.6.4 Buffer accessors
 */
template <typename DataType, int dimensions>
class accessor_buffer {
 protected:
  buffer<DataType, dimensions>* buf;
  handler* commandGroupHandler;
  range<dimensions> offset;
  range<dimensions> rang;

 public:
  accessor_buffer(buffer<DataType, dimensions>& bufferRef,
                  handler* commandGroupHandler, range<dimensions> offset,
                  range<dimensions> range)
      : buf(&bufferRef),
        commandGroupHandler(commandGroupHandler),
        offset(offset),
        rang(range) {}

 protected:
  cl_mem get_buffer_object() const {
    return buf->device_data.get();
  }
  ::size_t access_buffer_range(int n) const {
    return buf->rang.get(n);
  }
  typename base_host_data<DataType>::type* access_host_data() const {
    return buf->host_data.get();
  }
};

}  // namespace detail
}  // namespace sycl
}  // namespace cl
