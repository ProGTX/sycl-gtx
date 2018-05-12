#pragma once

namespace cl {
namespace sycl {
namespace detail {
namespace kernel_ns {

// Forward declaration
template <typename DataType, int dimensions, access::mode mode,
          access::target target>
static string_class register_resource(
    const accessor_core<DataType, dimensions, mode, target>& acc);

}  // namespace kernel_ns
}  // namespace detail
}  // namespace sycl
}  // namespace cl
