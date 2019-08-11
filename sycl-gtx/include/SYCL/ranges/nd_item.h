#pragma once

// 3.5.1.5 nd_item class

#include "SYCL/access.h"
#include "SYCL/detail/data_ref.h"
#include "SYCL/detail/point_ref.h"
#include "SYCL/ranges/point.h"

namespace cl {
namespace sycl {

// Forward declarations
template <int dimensions>
struct id;
template <int dimensions>
struct range;
template <int dimensions>
struct nd_range;

namespace detail {
namespace kernel_ns {
template <class Input>
struct constructor;
}
}  // namespace detail

template <int dimensions = 1>
struct nd_item {
 protected:
  friend struct detail::kernel_ns::constructor<nd_item<dimensions>>;

  item<dimensions> global_item;
  item<dimensions> local_item;

  nd_item(item<dimensions> global_item, item<dimensions> local_item)
      : global_item(global_item), local_item(local_item) {}

  // A bit of a hack - to the outside it appears to conform to the specification
  using size_t = detail::point_ref<true>;

 public:
  operator item<dimensions>() {
    return global_item;
  }

  id<dimensions> get_global() const {
    return global_item.get();
  }
  size_t get_global(int dimension) const {
    return get_global().get(dimension);
  }
  size_t get_global_linear_id() const {
    return global_item.get_linear_id();
  }

  id<dimensions> get_local() const {
    return local_item.get();
  }
  size_t get_local(int dimension) const {
    return get_local().get(dimension);
  }
  size_t get_local_linear_id() const {
    return local_item.get_linear_id();
  }

  id<dimensions> get_group() const {
    // TODO(progtx):
    return local_item.get_offset();
  }
  size_t get_group(int dimension) const {
    return get_group()[dimension];
  }
  size_t get_group_linear_id() const {
    // TODO(progtx):
    return size_t{};
  }

  // TODO(progtx):
  id<dimensions> get_num_groups() const;
  int get_num_groups(int) const;

  range<dimensions> get_global_range() const {
    return global_item.get_range();
  }
  range<dimensions> get_local_range() const {
    return local_item.get_range();
  }
  id<dimensions> get_offset() const {
    return global_item.get_offset();
  }
  nd_range<dimensions> get_nd_range() const {
    return nd_range<dimensions>(get_global_range(), get_local_range(),
                                get_offset());
  }

  void barrier(
      access::fence_space flag = access::fence_space::global_and_local) const {
    string_class flag_string;

    switch (flag) {
      case access::fence_space::local_space:
        flag_string = "CLK_LOCAL_MEM_FENCE";
        break;
      case access::fence_space::global_space:
        flag_string = "CLK_GLOBAL_MEM_FENCE";
        break;
      case access::fence_space::global_and_local:
      default:
        flag_string = "CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE";
        break;
    }

    detail::kernel_add(string_class("barrier(") + flag_string + ")");
  }
};

}  // namespace sycl
}  // namespace cl
