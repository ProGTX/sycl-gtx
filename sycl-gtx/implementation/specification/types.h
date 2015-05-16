#pragma once

// 3.9.1 Description of the built-in types available for SYCL host and device

#include "../common.h"
#include "../data_ref.h"

namespace cl {
namespace sycl {

namespace detail {

template <typename dataT, int numElements>
class cl_type : public data_ref {
};

} // namespace detail

} // namespace sycl
} // namespace cl
