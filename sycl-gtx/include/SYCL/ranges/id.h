#pragma once

// 3.5.1.3 ID class

#include "SYCL/detail/common.h"
#include "SYCL/detail/data_ref.h"
#include "SYCL/ranges/item.h"
#include "SYCL/ranges/point.h"
#include "SYCL/ranges/range.h"
#include <initializer_list>

namespace cl {
namespace sycl {

// Forward declaration
template <int dims>
struct nd_item;

namespace detail {

// Forward declarations
namespace kernel_ns {
class source;
template <class Input>
struct constructor;
}  // namespace kernel_ns

}  // namespace detail

template <int dimensions = 1>
struct id;

template <>
struct id<1> : detail::point<1> {
  template <class Input>
  friend struct detail::kernel_ns::constructor;
  friend class detail::data_ref;

  id(::size_t x = 0) : point<1>(x, 0, 0) {}
  id(const range<1>& rangeSize) : id(rangeSize.get(0)) {}
  id(const item<1>& rhs) : id(rhs.get()) {}
};

template <>
struct id<2> : detail::point<2> {
  template <class Input>
  friend struct detail::kernel_ns::constructor;
  friend class detail::data_ref;

  id(::size_t x = 0, ::size_t y = 0) : point<2>(x, y, 0) {}
  id(const range<2>& rangeSize) : id(rangeSize.get(0), rangeSize.get(1)) {}
  id(const item<2>& rhs) : id(rhs.get()) {}
};

template <>
struct id<3> : detail::point<3> {
  template <class Input>
  friend struct detail::kernel_ns::constructor;
  friend class detail::data_ref;

  id(::size_t x = 0, ::size_t y = 0, ::size_t z = 0) : point<3>(x, y, z) {}
  id(const range<3>& rangeSize)
      : id(rangeSize.get(0), rangeSize.get(1), rangeSize.get(2)) {}
  id(const item<3>& rhs) : id(rhs.get()) {}
};

namespace detail {

template <int dimensions>
struct get_special_id {
  static id<dimensions> global() {
    auto i = id<dimensions>();
    i.set(data_ref::type_t::id_global);
    return i;
  }
  static id<dimensions> local() {
    auto i = id<dimensions>();
    i.set(data_ref::type_t::id_local);
    return i;
  }
};

}  // namespace detail

}  // namespace sycl
}  // namespace cl
