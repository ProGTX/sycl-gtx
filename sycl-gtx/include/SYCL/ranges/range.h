#pragma once

// 3.5.1.1 Range class

#include "SYCL/ranges/point.h"

namespace cl {
namespace sycl {

template <int dimensions = 1>
struct range;

template <>
struct range<1> : detail::point<1> {
  range(::size_t x) : point<1>(x, 0, 0) {}
  ::size_t size() const {
    return values[0];
  }
};

template <>
struct range<2> : detail::point<2> {
  range(::size_t x, ::size_t y) : point<2>(x, y, 0) {}
  ::size_t size() const {
    return values[0] * values[1];
  }
};

template <>
struct range<3> : detail::point<3> {
  range(::size_t x, ::size_t y, ::size_t z) : point<3>(x, y, z) {}
  ::size_t size() const {
    return values[0] * values[1] * values[2];
  }
};

namespace detail {

template <int dimensions>
inline range<dimensions> empty_range();
template <>
inline range<1> empty_range() {
  return range<1>(0);
}
template <>
inline range<2> empty_range() {
  return range<2>(0, 0);
}
template <>
inline range<3> empty_range() {
  return range<3>(0, 0, 0);
}

template <int dimensions>
struct get_special_range {
  static range<dimensions> global() {
    auto r = empty_range<dimensions>();
    r.set(data_ref::type_t::range_global);
    return r;
  }
  static range<dimensions> local() {
    auto r = empty_range<dimensions>();
    r.set(data_ref::type_t::range_local);
    return r;
  }
};

}  // namespace detail

}  // namespace sycl
}  // namespace cl
