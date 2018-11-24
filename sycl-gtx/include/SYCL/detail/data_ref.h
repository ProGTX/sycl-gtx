#pragma once

#include "SYCL/detail/common.h"
#include "SYCL/detail/debug.h"
#include <type_traits>

namespace cl {
namespace sycl {

// Forward declaration
template <int dimensions>
struct id;

namespace detail {

// Forward declaration
void kernel_add(string_class line);

/**
 * Data reference wrappers
 */
class data_ref {
 public:
  enum class type_t {
    general,
    numeric,
    id_global,
    id_local,
    range_global,
    range_local,
    expression,
  };

  static const string_class open_parenthesis;
  string_class name;
  type_t type;

  static string_class get_name(const data_ref& dref) {
    return dref.name;
  }

  template <typename T, typename std::enable_if<
                            std::is_arithmetic<T>::value>::type* = nullptr>
  static string_class get_name(const T& n) {
    return get_string<T>::get(n);
  }

  template <typename T,
            typename std::enable_if<std::is_enum<T>::value>::type* = nullptr>
  static string_class get_name(const T& n) {
    auto value = static_cast<typename std::underlying_type<T>::type>(n);
    return get_string<decltype(value)>::get(value);
  }

  data_ref(string_class name) : name(name) {}

  data_ref(char* name) : name(name) {}

  data_ref(const char* name) : name(name) {}

  template <class T>
  data_ref(T&& type) : name(get_name(type)) {}

  data_ref(const data_ref& copy) = default;
#if MSVC_2013_OR_LOWER
  data_ref(data_ref&& move) : SYCL_MOVE_INIT(name), SYCL_MOVE_INIT(type) {}
  friend void swap(data_ref& first, data_ref& second) {
    using std::swap;
    SYCL_SWAP(name);
    SYCL_SWAP(type);
  }
#else
  data_ref(data_ref&&) = default;
#endif
  ~data_ref() = default;

  // We need to generate a new line, no matter whether moving or copying
  data_ref& operator=(const data_ref& dref) {
    kernel_add(name + " = " + dref.name);
    return *this;
  }
  data_ref& operator=(data_ref&& dref) noexcept {
    kernel_add(name + " = " + dref.name);
    return *this;
  }

  // TODO(progtx):
  // https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/operators.html

#define SYCL_ASSIGNMENT_OPERATOR(op)              \
  template <class T>                              \
  data_ref& operator op(const T& n) {             \
    kernel_add(name + " " #op " " + get_name(n)); \
    return *this;                                 \
  }

#define SYCL_DATA_REF_OPERATOR(op)                                             \
  template <class T>                                                           \
  data_ref operator op(const T& n) const {                                     \
    return data_ref(open_parenthesis + name + " " #op " " + get_name(n) +      \
                    ')');                                                      \
  }                                                                            \
  template <typename T,                                                        \
            typename std::enable_if<std::is_arithmetic<T>::value>::type* =     \
                nullptr>                                                       \
  friend data_ref operator op(const T& n, const data_ref& dref) {              \
    return data_ref(open_parenthesis + get_name(n) + " " #op " " + dref.name + \
                    ')');                                                      \
  }

  SYCL_ASSIGNMENT_OPERATOR(=);

  // Arithmetic operatos
  SYCL_DATA_REF_OPERATOR(+);
  SYCL_ASSIGNMENT_OPERATOR(+=);
  SYCL_DATA_REF_OPERATOR(-);
  SYCL_ASSIGNMENT_OPERATOR(-=);
  SYCL_DATA_REF_OPERATOR(*);
  SYCL_ASSIGNMENT_OPERATOR(*=);
  SYCL_DATA_REF_OPERATOR(/);
  SYCL_ASSIGNMENT_OPERATOR(/=);
  SYCL_DATA_REF_OPERATOR(%);
  SYCL_ASSIGNMENT_OPERATOR(%=);

  // Comparison operators
  SYCL_DATA_REF_OPERATOR(==);
  SYCL_DATA_REF_OPERATOR(!=);
  SYCL_DATA_REF_OPERATOR(<);
  SYCL_DATA_REF_OPERATOR(<=);
  SYCL_DATA_REF_OPERATOR(>);
  SYCL_DATA_REF_OPERATOR(>=);

  // Boolean operators
  SYCL_DATA_REF_OPERATOR(||);
  SYCL_DATA_REF_OPERATOR(&&);

  // Bit operators
  SYCL_DATA_REF_OPERATOR(&);
  SYCL_ASSIGNMENT_OPERATOR(&=);
  SYCL_DATA_REF_OPERATOR(|);
  SYCL_ASSIGNMENT_OPERATOR(|=);
  SYCL_DATA_REF_OPERATOR (^);
  SYCL_ASSIGNMENT_OPERATOR(^=);

  // Shifts
  SYCL_DATA_REF_OPERATOR(>>);
  SYCL_ASSIGNMENT_OPERATOR(>>=);
  SYCL_DATA_REF_OPERATOR(<<);
  SYCL_ASSIGNMENT_OPERATOR(<<=);

#undef SYCL_ASSIGNMENT_OPERATOR
#undef SYCL_DATA_REF_OPERATOR

  // TODO(progtx): Increment and decrement can be either a statement or an
  // expression
  // But there is no way to distinguish it
  // Here presume an expression
  data_ref operator++() const {
    return data_ref(open_parenthesis + "++" + name + ')');
  }
  data_ref operator++(int) const {
    return data_ref(open_parenthesis + name + "++" + ')');
  }
  data_ref operator--() const {
    return data_ref(open_parenthesis + "--" + name + ')');
  }
  data_ref operator--(int) const {
    return data_ref(open_parenthesis + name + "--" + ')');
  }

  data_ref operator!() const {
    return data_ref(open_parenthesis + '!' + name + ')');
  }
};

}  // namespace detail

}  // namespace sycl
}  // namespace cl
