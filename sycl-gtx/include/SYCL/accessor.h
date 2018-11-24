#pragma once

// 3.4.6 Accessors

#include "SYCL/access.h"
#include "SYCL/detail/common.h"

namespace cl {
namespace sycl {

namespace detail {

// Forward declaration
namespace kernel_ns {
class source;
}

class accessor_base {};

// 3.6.4.3 Core accessors class
template <typename DataType, int dimensions, access::mode mode,
          access::target target>
class accessor_core : public accessor_base {
 public:
  using value_type = DataType;
  using reference = value_type&;
  using const_reference = const value_type&;

  /** Returns the size of the underlying buffer in number of elements. */
  ::size_t get_size() const;

  // TODO(progtx): Only available when target is cl_image or cl_buffer
  /** @return the cl_mem object corresponding to the access. */
  virtual cl_mem get_cl_mem_object() const {
    return nullptr;
  }

  // TODO(progtx): Only available when target is cl_image or cl_buffer.
  /** @return the cl_event object corresponding to the last command to access
   * the memory object. */
  cl_event get_cl_event_object() const;

  virtual ~accessor_core() = default;

 protected:
  friend class kernel_ns::source;

  virtual void* resource() const {
    return nullptr;
  }

  virtual ::size_t argument_size() const {
    return 0;
  }
};

template <bool>
struct select_target;

template <typename DataType, int dimensions, access::mode mode,
          access::target target, typename = select_target<true>>
class accessor_detail;

#define SYCL_ACCESSOR_CLASS(condition)                            \
  template <typename DataType, int dimensions, access::mode mode, \
            access::target target>                                \
  class accessor_detail<DataType, dimensions, mode, target,       \
                        select_target<(condition)>>               \
      : public accessor_core<DataType, dimensions, mode, target>

}  // namespace detail

template <typename DataType, int dimensions = 1,
          access::mode mode = access::mode::read_write,
          access::target target = access::target::global_buffer>
class accessor;

#define SYCL_ADD_ACCESSOR(mode, target)              \
  template <typename DataType, int dimensions>       \
  class accessor<DataType, dimensions, mode, target> \
      : public detail::accessor_detail<DataType, dimensions, mode, target>

}  // namespace sycl
}  // namespace cl
