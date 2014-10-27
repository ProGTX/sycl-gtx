#pragma once

// 3.3.1 Buffers

#include <vector>

namespace cl {
namespace sycl {

// typename T: The type of the elements of the buffer
// int dimensions: number of dimensions of the buffer : 1, 2, or 3
template <typename T, int dimensions>
struct buffer {};

} // namespace sycl
} // namespace cl
