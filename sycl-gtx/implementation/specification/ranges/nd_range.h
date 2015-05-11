#pragma once

// 3.7.1.2 nd_range class

#include "id.h"
#include "range.h"

namespace cl {
namespace sycl {

template <int dims = 1>
struct nd_range {
private:
	range<dims> global_size;
	range<dims> local_size;
	id<dims> offset;

public:
	static_assert(1 <= dims && dims <= 3, "Dimensions are between 1 and 3");

	nd_range(
		range<dims> global_size,
		range<dims> local_size,
		id<dims> offset = id<dims>()
	)
		: global_size(global_size), local_size(local_size), offset(offset) {}

	range<dims> get_global_range() const {
		return global_size;
	}
	
	range<dims> get_local_range() const {
		return local_size;
	}

	// Return a range representing the number of groups in each dimension.
	range<dims> get_group_range() const {
		return global_size / local_size;
	}

	id<dims> get_offset() const {
		return offset;
	}
};

} // namespace sycl
} // namespace cl
