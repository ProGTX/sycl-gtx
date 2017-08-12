#pragma once

#include "SYCL/detail/common.h"

namespace cl {
namespace sycl {
namespace detail {

using counter_t = unsigned int;

template <class T, counter_t start = 0>
class counter {
 private:
  static counter_t internal_count;
  counter_t counter_id;

 public:
  counter() : counter_id(internal_count++) {}

  counter(const counter& copy) : counter() {}
  counter(counter&& move) noexcept : counter_id(move.counter_id) {}
  counter& operator=(const counter& copy) {
    counter_id = copy.counter_id;
    return *this;
  }
  counter& operator=(counter&& move) noexcept {
    return *this;
  }
  ~counter() = default;

  static counter_t get_total_count() {
    return internal_count;
  }

  counter_t get_count_id() const {
    return counter_id;
  }
};

template <class T, counter_t start>
counter_t counter<T, start>::internal_count = start;

}  // namespace detail
}  // namespace sycl
}  // namespace cl
