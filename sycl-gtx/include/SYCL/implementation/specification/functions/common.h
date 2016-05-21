#pragma once

// 3.9.5 Common Functions

#include "../vectors/vec.h"
#include "../../data_ref.h"

namespace cl {
namespace sycl {

#define SYCL_ONE_ARG(name)                          \
template <class First>                              \
static detail::data_ref name(const First& first) {  \
  using detail::data_ref;                           \
  return  data_ref(string_class(#name "(") +        \
      data_ref::get_name(first) + ')');             \
}

SYCL_ONE_ARG(cos)
SYCL_ONE_ARG(fabs)
SYCL_ONE_ARG(sin)
SYCL_ONE_ARG(sqrt)

#undef SYCL_ONE_ARG

template <class First, class Second>
static detail::data_ref min(const First& first, const Second& second) {
  using detail::data_ref;
  return data_ref(
    string_class("min(") +
    data_ref::get_name(first) + ", " + data_ref::get_name(second) +
    ")"
  );
}

} // namespace sycl
} // namespace cl
