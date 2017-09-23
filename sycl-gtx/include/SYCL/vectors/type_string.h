#pragma once

#include "SYCL/detail/common.h"
#include "SYCL/vectors/cl_type.h"
#include "SYCL/vectors/vec.h"

namespace cl {
namespace sycl {

namespace detail {

/**
 * Cannot be joined with vector declarations
 * because a vector of 3 is a typedef of a vector of 4
 */
#define SYCL_CL_TYPE_STRING(nummedType)             \
  template <>                                       \
  struct type_string<::cl_##nummedType> {           \
    static string_class get() {                     \
      return #nummedType;                           \
    }                                               \
  };                                                \
  template <>                                       \
  struct type_string<::cl::sycl::cl_##nummedType> { \
    static string_class get() {                     \
      return #nummedType;                           \
    }                                               \
  };

#define SYCL_ADD_CL_TYPE_STRING(basetype) \
  SYCL_CL_TYPE_STRING(basetype##2)        \
  SYCL_CL_TYPE_STRING(basetype##4)        \
  SYCL_CL_TYPE_STRING(basetype##8)        \
  SYCL_CL_TYPE_STRING(basetype##16)

SYCL_ADD_CL_TYPE_STRING(int)
SYCL_ADD_CL_TYPE_STRING(char)
SYCL_ADD_CL_TYPE_STRING(short)
SYCL_ADD_CL_TYPE_STRING(long)
SYCL_ADD_CL_TYPE_STRING(float)
SYCL_ADD_CL_TYPE_STRING(double)

SYCL_ADD_CL_TYPE_STRING(uint)
SYCL_ADD_CL_TYPE_STRING(uchar)
SYCL_ADD_CL_TYPE_STRING(ushort)
SYCL_ADD_CL_TYPE_STRING(ulong)

#undef SYCL_ADD_CL_TYPE_STRING
#undef SYCL_CL_TYPE_STRING

}  // namespace detail

}  // namespace sycl
}  // namespace cl
