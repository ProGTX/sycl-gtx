#pragma once

#include "SYCL/detail/common.h"
#include "SYCL/vectors/cl_vec.h"

namespace cl {
namespace sycl {

// Forward declaration
template <typename dataT, int numElements>
class vec;

namespace detail {

// These are defined elsewhere, here only specialization for vectors

template <typename dataT, int numElements>
struct data_size<vec<dataT, numElements>> {
  static ::size_t get() {
    return sizeof(typename base_host_data<vec<dataT, numElements>>::type);
  }
};

template <typename dataT, int numElements>
struct base_host_data<vec<dataT, numElements>> {
  using type = vectors::cl_base<dataT, numElements, numElements>;
};

template <typename dataT, int numElements>
struct acc_device_return<vec<dataT, numElements>> {
  using type = vec<dataT, numElements>;
};
template <typename dataT, int numElements>
struct acc_device_return<vectors::cl_base<dataT, numElements, numElements>> {
  using type = vec<dataT, numElements>;
};

}  // namespace detail

}  // namespace sycl
}  // namespace cl
