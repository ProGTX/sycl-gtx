#pragma once

#include "SYCL/ranges/id.h"
#include "SYCL/ranges/range.h"

namespace cl {
namespace sycl {

/**
 * 3.5.1.2 nd_range class
 */
template <int dims = 1>
struct nd_range {
 private:
  range<dims> global_size;
  range<dims> local_size;
  id<dims> offset;

 public:
  static_assert(1 <= dims && dims <= 3, "Dimensions are between 1 and 3");

  nd_range(range<dims> global_size, range<dims> local_size,
           id<dims> offset = id<dims>())
      : global_size(global_size), local_size(local_size), offset(offset) {}

  range<dims> get_global() const {
    return global_size;
  }

  range<dims> get_local() const {
    return local_size;
  }

  // Return a range representing the number of groups in each dimension.
  range<dims> get_group() const {
    return global_size / local_size;
  }

  id<dims> get_offset() const {
    return offset;
  }
};

}  // namespace sycl
}  // namespace cl
