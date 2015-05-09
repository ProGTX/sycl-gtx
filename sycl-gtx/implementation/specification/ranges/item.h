#pragma once

// 3.7.1.4 Item class

namespace cl {
namespace sycl {

// Forward declarations
template <int dims>
struct id;
template <int dims>
struct range;

template <int dims = 1>
struct item {
	item() = delete;
	id<dims> get_global_id() const;

	size_t get(int dimension) const;
	size_t& operator[](int dimension);

	range<dims> get_global_range() const;
	id<dims> get_offset() const;
};

} // namespace sycl
} // namespace cl
