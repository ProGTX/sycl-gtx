#pragma once

// 3.7.1.2 nd_range class

namespace cl {
namespace sycl {

// Forward declarations
template <int dims>
struct id;
template <int dims>
struct range;

template <int dims = 1>
struct nd_range {
	static_assert(1 <= dims && dims <= 3, "Dimensions are between 1 and 3");

	nd_range(
		range<dims> global_size,
		range<dims> local_size,
		id<dims> offset = id<dims>()
	);

	range<dims> get_global_range() const;
	range<dims> get_local_range() const;
	range<dims> get_group_range() const;

	id<dims> get_offset() const;
};

} // namespace sycl
} // namespace cl
