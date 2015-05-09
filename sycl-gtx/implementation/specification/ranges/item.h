#pragma once

// 3.7.1.4 Item class

namespace cl {
namespace sycl {

// Forward declarations
template <int dims>
struct id;
template <int dims>
struct range;
template <class Input>
struct constructor;

template <int dims = 1>
struct item {
private:
	friend struct constructor<item<dims>>;

	id<dims> global_id;
	range<dims> global_range;
	id<dims> offset;

	item(id<dims> global_id, range<dims> global_range, id<dims> offset)
		: global_id(global_id), global_range(global_range), offset(offset) {}
public:
	item() = delete;
	id<dims> get_global_id() const;

	size_t get(int dimension) const;
	size_t& operator[](int dimension);

	range<dims> get_global_range() const;
	id<dims> get_offset() const;
};

} // namespace sycl
} // namespace cl
