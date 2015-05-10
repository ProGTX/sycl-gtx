#pragma once

// 3.7.1.5 nd_item class

#include "../access.h"
#include "item.h"

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
struct nd_item : public item<dims> {
protected:
	nd_item(id<dims> global_id, range<dims> global_range, id<dims> offset = id<dims>())
		: item(global_id, global_range, offset) {}
public:
	id<dims> get_global_id() const {
		return item::get_global_id();
	}
	size_t get_global_id(int dimension) const;

	id<dims> get_local_id() const;
	size_t get_local_id(int dimension) const;

	id<dims> get_group_id() const;
	size_t get_group_id(int dimension) const;

	range<dims> get_global_range() const {
		return item::get_global_range();
	}
	range<dims> get_local_range() const;
	id<dims> get_offset() const {
		return item::get_offset();
	}
	nd_range<dims> get_nd_range() const;

	void barrier(access::fence_space flag = access::fence_space::global_and_local) const;

	// Remains of the item class
	size_t get(int dimension) const = delete;
	size_t& operator[](int dimension) = delete;
};

} // namespace sycl
} // namespace cl
