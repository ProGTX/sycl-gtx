#pragma once

// 3.7.2 Vector types
// Swizzled intermediate vectors

#include "../../data_ref.h"

namespace cl {
namespace sycl {

// Forward declaration
template <typename dataT, int numElements>
class vec;


template <typename dataT, int numElements>
class swizzled_vec : public detail::data_ref {};


} // namespace sycl
} // namespace cl
