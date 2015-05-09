#pragma once

// 3.7.1.5 nd_item class

#include "../access.h"

namespace cl {
namespace sycl {

// Forward declarations
template <int dims>
struct id;
template <int dims>
struct range;
template <int dims>
struct nd_range;

template <int dims = 1>
struct nd_item {
	nd_item() = delete;

	id<dims> get_global_id() const;
	size_t get_global_id(int dimension) const;

	id<dims> get_local_id() const;
	size_t get_local_id(int dimension) const;

	id<dims> get_group_id() const;
	size_t get_group_id(int dimension) const;

	range<dims> get_global_range() const;
	range<dims> get_local_range() const;
	id<dims> get_offset() const;
	nd_range<dims> get_nd_range() const;

	void barrier(access::fence_space flag = access::global_and_local) const;
};

} // namespace sycl
} // namespace cl
