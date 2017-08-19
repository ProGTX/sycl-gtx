#pragma once

// 3.7.2 Vector types
// Swizzled intermediate vectors

namespace cl {
namespace sycl {

namespace detail {
namespace vectors {

template <int index, int current>
static void to_char(char* name) {
  name[index] = static_cast<char>(current) + '0';
}

template <int index, int current, int... others>
struct swizzled;

template <int index, int current>
struct swizzled<index, current> {
  static void get(char* name) {
    to_char<index, current>(name);
  }
};

template <int index, int current, int... others>
struct swizzled {
  static void get(char* name) {
    to_char<index, current>(name);
    swizzled<index + 1, others...>::get(name);
  }
};

}  // namespace vectors
}  // namespace detail

}  // namespace sycl
}  // namespace cl
