#pragma once

// 3.5.1.4 Item class

#include "SYCL/detail/point_ref.h"

namespace cl {
namespace sycl {

// Forward declarations
template <int dimensions>
struct id;
template <int dimensions>
struct range;
template <int dimensions>
struct nd_item;

namespace detail {

namespace kernel_ns {
// Forward declaration
template <class Input>
struct constructor;
}  // namespace kernel_ns
}  // namespace detail

template <int dimensions = 1>
struct item {
 private:
  id<dimensions> index;
  range<dimensions> rang;
  id<dimensions> offset;

 protected:
  friend struct detail::kernel_ns::constructor<item<dimensions>>;
  friend struct detail::kernel_ns::constructor<nd_item<dimensions>>;

  item(id<dimensions> global_id, range<dimensions> global_range,
       id<dimensions> offset = id<dimensions>())
      : index(global_id), rang(global_range), offset(offset) {}

  using size_t_cref = detail::point_ref<true>;
  using size_t_ref = detail::point_ref<false>;

 public:
  id<dimensions> get() const {
    return index;
  }
  range<dimensions> get_range() const {
    return rang;
  }
  id<dimensions> get_offset() const {
    return offset;
  }

  size_t_cref get(int dimension) const {
    return index.get(dimension);
  }
  size_t_ref operator[](int dimension) {
    return index[dimension];
  }

  // TODO(progtx): Return the linearized ID in the item's range.
  /** Computed as the flatted ID after the offset is subtracted. */
  ::size_t get_linear_id() const {
    return 0;
  }

  operator id<dimensions>() {
    return index;
  }
};

}  // namespace sycl
}  // namespace cl
