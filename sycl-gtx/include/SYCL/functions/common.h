#pragma once

// 3.9.5 Common Functions

#include "SYCL/detail/data_ref.h"
#include "SYCL/vectors/vec.h"

namespace cl {
namespace sycl {

#define SYCL_ONE_ARG(NAME)                                                \
  template <class First>                                                  \
  static detail::data_ref NAME(const First& first) {                      \
    using detail::data_ref;                                               \
    return data_ref(string_class(#NAME "(") + data_ref::get_name(first) + \
                    ')');                                                 \
  }

SYCL_ONE_ARG(cos);
SYCL_ONE_ARG(fabs);
SYCL_ONE_ARG(sin);
SYCL_ONE_ARG(sqrt);

#undef SYCL_ONE_ARG

#define SYCL_TWO_ARG(NAME)                                                 \
  template <class First, class Second>                                     \
  static detail::data_ref NAME(const First& first, const Second& second) { \
    using detail::data_ref;                                                \
    return data_ref(string_class(#NAME "(") + data_ref::get_name(first) +  \
                    ", " + data_ref::get_name(second) + ")");              \
  }

SYCL_TWO_ARG(min);
SYCL_TWO_ARG(pow);

#undef SYCL_TWO_ARG

}  // namespace sycl
}  // namespace cl
