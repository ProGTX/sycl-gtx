#pragma once

#include <type_traits>

namespace cl {
namespace sycl {
namespace detail {

template <typename T, bool holds_pointer = false>
struct ptr_or_val;

template <typename T>
struct ptr_or_val<T, false> {
 private:
  friend struct ptr_or_val<T, true>;
  T data;

 public:
  ptr_or_val(T value = 0) : data(value) {}
  ptr_or_val(const ptr_or_val<T, true>& ptr) : data(*(ptr.data)) {}

  ptr_or_val& operator=(T n) {
    this->data = n;
    return *this;
  }

  operator T() const {
    return this->data;
  }
  operator T&() {
    return this->data;
  }

  ptr_or_val<T, true> operator&() {  // NOLINT
    return ptr_or_val<T, true>(&this->data);
  }
  ptr_or_val<typename std::remove_pointer<T>::type, false> operator*() {
    return ptr_or_val<typename std::remove_pointer<T>::type, false>(
        *this->data);
  }
};

template <typename T>
struct ptr_or_val<T, true> {
 private:
  friend struct ptr_or_val<T, false>;
  T* data;

 public:
  ptr_or_val(T* ptr) : data(ptr) {}
  ptr_or_val(ptr_or_val<T, false>& value) : data(&(value.data)) {}

  ptr_or_val& operator=(T n) {
    *this->data = n;
    return *this;
  }

  operator T() const {
    return *this->data;
  }
  operator T&() {
    return *this->data;
  }

  ptr_or_val<T*, true> operator&() {  // NOLINT
    return ptr_or_val<T*, true>(&this->data);
  }
  ptr_or_val<typename std::remove_pointer<T>::type, false> operator*() {
    return ptr_or_val<typename std::remove_pointer<T>::type, false>(
        *this->data);
  }
};

}  // namespace detail
}  // namespace sycl
}  // namespace cl
