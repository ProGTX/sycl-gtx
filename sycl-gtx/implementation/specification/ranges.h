#pragma once

// 3.7.1 Ranges and identifiers

#include "ranges\range.h"
#include "ranges\id.h"
#include "../common.h"
#include "../debug.h"
#include <initializer_list>

namespace cl {
namespace sycl {

// 3.7.1.2 nd range class
template <int dimensions = 1>
class nd_range {
private:
	range<dimensions> global_size;
	range<dimensions> local_size;

	// TODO
	id<dimensions> offset;

public:
	static_assert(1 <= dimensions && dimensions <= 3, "Dimensions are between 1 and 3");

	nd_range(range<dimensions> global_size, range<dimensions> local_size, id<dimensions> offset = id<dimensions>())
		: global_size(global_size), local_size(local_size), offset(offset) {}

	range<dimensions> get_global_range() const {
		return global_size;
	}
	range<dimensions> get_local_range() const {
		return local_size;
	}
	id<dimensions> get_offset() const {
		return offset;
	}

	// Returns a range representing the number of groups in each dimension.
	range<dimensions> get_group_range() const {
		return global_size / local_size;
	}
};


// TODO: 3.7.1.6 Group class


} // namespace sycl
} // namespace cl
