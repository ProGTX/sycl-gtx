#pragma once

// 3.7.1.4 Item class

namespace cl {
namespace sycl {

// Forward declarations
template <int dims>
struct id;
template <int dims>
struct range;
template <int dims>
struct nd_item;

namespace detail {
namespace kernel_ {
	template <class Input>
	struct constructor;
}
}

template <int dims = 1>
struct item {
private:
	id<dims> index;
	range<dims> rang;
	id<dims> offset;

protected:
	friend struct detail::kernel_::constructor<item<dims>>;
	friend struct detail::kernel_::constructor<nd_item<dims>>;

	item(id<dims> global_id, range<dims> global_range, id<dims> offset = id<dims>())
		: index(global_id), rang(global_range), offset(offset) {}
public:
	id<dims> get_global_id() const {
		return index;
	}
	range<dims> get_global_range() const {
		return rang;
	}
	id<dims> get_offset() const {
		return offset;
	}

	size_t get(int dimension) const;
	size_t& operator[](int dimension);
};

} // namespace sycl
} // namespace cl
