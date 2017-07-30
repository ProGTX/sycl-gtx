#pragma once

#include "SYCL/detail/debug.h"

namespace cl {
namespace sycl {
namespace access {

// 3.8 Synchronization and atomics
enum class fence_space : char { local_space, global_space, global_and_local };

// 3.4.6.1 Access modes
enum class mode {
  /** read-only access */
  read,
  /** write-only access, previous contents NOT discarded */
  write,
  /** read and write access */
  read_write,
  /** write-only access, previous contents discarded */
  discard_write,
  /** read and write access, previous contents discarded */
  discard_read_write,
  /** atomic access */
  atomic
};

// 3.4.6.2 Access targets
enum class target {
  /** access buffer via __global memory */
  global_buffer,
  /** access buffer via __constant memory */
  constant_buffer,
  /** access work-group-local memory */
  local,
  /** access an image */
  image,
  /** access buffer immediately on the host */
  host_buffer,
  /** access image immediately on the host */
  host_image,
  /** access an array of images on device */
  image_array
};

static debug& operator<<(debug& d, mode m) {
  std::string str("mode::");
  switch (m) {
    case mode::read:
      str += "read";
      break;
    case mode::write:
      str += "write";
      break;
    case mode::read_write:
      str += "read_write";
      break;
    case mode::discard_write:
      str += "discard_write";
      break;
    case mode::discard_read_write:
      str += "discard_read_write";
      break;
    case mode::atomic:
      str += "atomic";
      break;
  }
  d << str;
  return d;
}

static debug& operator<<(debug& d, target t) {
  std::string str("target::");
  switch (t) {
    case target::global_buffer:
      str += "global_buffer";
      break;
    case target::constant_buffer:
      str += "constant_buffer";
      break;
    case target::local:
      str += "local";
      break;
    case target::image:
      str += "image";
      break;
    case target::host_buffer:
      str += "host_buffer";
      break;
    case target::host_image:
      str += "host_image";
      break;
    case target::image_array:
      str += "image_array";
      break;
  }
  d << str;
  return d;
}

}  // namespace access

namespace detail {

// Forward declaration
class buffer_base;

struct buffer_access {
  buffer_base* data;
  access::mode mode;
  access::target target;
};

}  // namespace detail

}  // namespace sycl
}  // namespace cl
