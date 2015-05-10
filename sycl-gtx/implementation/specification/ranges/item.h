#pragma once

// 3.7.1.4 Item class

namespace cl {
namespace sycl {

// Forward declarations
template <int dims>
struct id;
template <int dims>
struct range;

namespace detail {
namespace kernel_ {
	template <class Input>
	struct constructor;
}
}

template <int dims = 1>
struct item {
protected:
	friend struct detail::kernel_::constructor<item<dims>>;

	id<dims> global_id;
	range<dims> global_range;
	id<dims> offset;

	item(id<dims> global_id, range<dims> global_range, id<dims> offset = id<dims>())
		: global_id(global_id), global_range(global_range), offset(offset) {}
public:
	id<dims> get_global_id() const {
		return global_id;
	}
	range<dims> get_global_range() const {
		return global_range;
	}
	id<dims> get_offset() const {
		return offset;
	}

	size_t get(int dimension) const;
	size_t& operator[](int dimension);
};

} // namespace sycl
} // namespace cl
