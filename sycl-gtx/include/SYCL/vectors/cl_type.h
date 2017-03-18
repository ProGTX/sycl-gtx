#pragma once

#include "SYCL/detail/common.h"

namespace cl {
namespace sycl {

namespace detail {

#define SYCL_CL_SCALAR(base)  \
  template <>                 \
  struct cl_type<base, 1> {   \
    using type = ::cl_##base; \
  };

#define SYCL_CL_USCALAR(base)        \
  SYCL_CL_SCALAR(base)               \
  template <>                        \
  struct cl_type<unsigned base, 1> { \
    using type = ::cl_u##base;       \
  };

#define SYCL_CL_VECTOR(base, num)  \
  template <>                      \
  struct cl_type<base, num> {      \
    using type = ::cl_##base##num; \
  };

#define SYCL_CL_UVECTOR(base, num)     \
  SYCL_CL_VECTOR(base, num)            \
  template <>                          \
  struct cl_type<unsigned base, num> { \
    using type = ::cl_u##base##num;    \
  };

#define SYCL_ADD_CL_VECTOR(base) \
  SYCL_CL_SCALAR(base)           \
  SYCL_CL_VECTOR(base, 2)        \
  SYCL_CL_VECTOR(base, 3)        \
  SYCL_CL_VECTOR(base, 4)        \
  SYCL_CL_VECTOR(base, 8)        \
  SYCL_CL_VECTOR(base, 16)

#define SYCL_ADD_CL_UVECTOR(base) \
  SYCL_CL_USCALAR(base)           \
  SYCL_CL_UVECTOR(base, 2)        \
  SYCL_CL_UVECTOR(base, 3)        \
  SYCL_CL_UVECTOR(base, 4)        \
  SYCL_CL_UVECTOR(base, 8)        \
  SYCL_CL_UVECTOR(base, 16)

SYCL_CL_SCALAR(bool)
SYCL_ADD_CL_UVECTOR(int)
SYCL_ADD_CL_UVECTOR(char)
SYCL_ADD_CL_UVECTOR(short)
SYCL_ADD_CL_UVECTOR(long)
SYCL_ADD_CL_VECTOR(float)
SYCL_ADD_CL_VECTOR(double)

#undef SYCL_CL_SCALAR
#undef SYCL_CL_USCALAR
#undef SYCL_CL_VECTOR
#undef SYCL_CL_UVECTOR
#undef SYCL_ADD_CL_VECTOR
#undef SYCL_ADD_CL_UVECTOR

}  // namespace detail

}  // namespace sycl
}  // namespace cl
