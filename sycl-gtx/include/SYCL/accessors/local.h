#pragma once

// 3.6.4.6 Local accessors

#include "SYCL/access.h"
#include "SYCL/accessor.h"
#include "SYCL/accessors/device_reference.h"

#include "SYCL/buffer.h"
#include "SYCL/command_group.h"
#include "SYCL/detail/common.h"
#include "SYCL/detail/counter.h"
#include "SYCL/ranges.h"

namespace cl {
namespace sycl {

namespace detail {

SYCL_ACCESSOR_CLASS(target == access::target::local)
, protected counter<accessor_detail<DataType, dimensions, mode, target>>,
    public accessor_device_ref<dimensions, DataType, dimensions, mode, target> {
 private:
  using base_acc_device_ref =
      accessor_device_ref<dimensions, DataType, dimensions, mode, target>;

 protected:
  template <int level, typename, int, access::mode, access::target>
  friend class accessor_device_ref;

  range<dimensions> allocationSize;

  ::size_t access_buffer_range(int n) const {
    return allocationSize.get(n);
  }

  void* resource() const final {
    return reinterpret_cast<void*>(this->get_count_id());  // NOLINT
  }

  ::size_t argument_size() const final {
    return data_size<DataType>::get() * allocationSize.size();
  }

 public:
  accessor_detail(range<dimensions> allocationSize,
                  handler & commandGroupHandler)
      : base_acc_device_ref(this, {}), allocationSize(allocationSize) {
    // TODO(progtx):
    if (command::group_detail::in_scope()) {
      command::group_detail::add_buffer_access(
          buffer_access{nullptr, mode, target}, __func__);
    }
  }

 private:
  using subscript_return_t =
      typename subscript_helper<dimensions, DataType, dimensions, mode,
                                target>::type;

 public:
  SYCL_DEVICE_REF_SUBSCRIPT_OPERATORS(base_acc_device_ref::);
};

}  // namespace detail

#if MSVC_2013_OR_LOWER
#define SYCL_ADD_ACCESSOR_LOCAL(mode)                                    \
  SYCL_ADD_ACCESSOR(mode, access::target::local) {                       \
    using Base = detail::accessor_detail<DataType, dimensions, mode,     \
                                         access::target::local>;         \
                                                                         \
   public:                                                               \
    accessor(range<dimensions> allocationSize) : Base(allocationSize) {} \
  };
#else
#define SYCL_ADD_ACCESSOR_LOCAL(mode)                                \
  SYCL_ADD_ACCESSOR(mode, access::target::local) {                   \
    using Base = detail::accessor_detail<DataType, dimensions, mode, \
                                         access::target::local>;     \
                                                                     \
   public:                                                           \
    using Base::Base;                                                \
  };
#endif

// 3.6.4.9 Accessor capabilities and restrictions
SYCL_ADD_ACCESSOR_LOCAL(access::mode::read)
SYCL_ADD_ACCESSOR_LOCAL(access::mode::write)
SYCL_ADD_ACCESSOR_LOCAL(access::mode::read_write)

}  // namespace sycl
}  // namespace cl
